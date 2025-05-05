import logging
from typing import Callable, Optional, Any, Dict, List, Literal, Tuple
# import random
# import textwrap
# import time
# import sys
# import select
# import numpy as np
# import optuna
# from optuna.distributions import CategoricalDistribution
# from collections import defaultdict

import dspy

# from dspy.propose.dataset_summary_generator import create_dataset_summary
from dspy.propose.utils import create_example_string, create_predictor_level_history_string, strip_prefix, get_dspy_source_code
from dspy.teleprompt.utils import get_signature
# from dspy.teleprompt.utils import get_prompt_model
# from dspy.propose.utils import get_dspy_source_code
# from dspy.propose.propose_base import Proposer
from dspy.propose.grounded_proposer import DescribeProgram, DescribeModule, TIPS
from dspy.propose.grounded_proposer import GroundedProposer
from dspy.propose.utils import create_predictor_level_history_string, strip_prefix

from dspy.teleprompt import MIPROv2
from dspy.propose import GroundedProposer

from dspy.teleprompt.utils import (
    # create_minibatch,
    # create_n_fewshot_demo_sets,
    # eval_candidate_program,
    # get_program_with_highest_avg_score,
    get_signature,
    # print_full_program,
    # save_candidate_program,
    # set_signature,
)


logger = logging.getLogger(__name__)

# Constants (copied from grounded_proposer.py)
MAX_INSTRUCT_IN_HISTORY = 5

# Constants from MIPROv2 (copied from mipro_optimizer_v2.py)
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
MIN_MINIBATCH_SIZE = 50
AUTO_RUN_SETTINGS = {
    "light": {"n": 6, "val_size": 100},
    "medium": {"n": 12, "val_size": 300},
    "heavy": {"n": 18, "val_size": 1000},
}
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"


# --- Custom Signature with Knowledge ---
def generate_instruction_class_with_knowledge(
    use_dataset_summary=True,
    program_aware=True,
    use_task_demos=True,
    use_instruct_history=True,
    use_tip=True,
):
    class GenerateSingleModuleInstructionWithKnowledge(dspy.Signature):
        """Use the information below, including the custom knowledge document, to learn about a task that we are trying to solve using calls to an LM, then generate a new instruction that will be used to prompt a Language Model to better solve the task."""

        if use_dataset_summary:
            dataset_description = dspy.InputField(
                desc="A description of the dataset that we are using.",
                prefix="DATASET SUMMARY:",
            )
        if program_aware:
            program_code = dspy.InputField(
                format=str,
                desc="Language model program designed to solve a particular task.",
                prefix="PROGRAM CODE:",
            )
            program_description = dspy.InputField(
                desc="Summary of the task the program is designed to solve, and how it goes about solving it.",
                prefix="PROGRAM DESCRIPTION:",
            )
            module = dspy.InputField(
                desc="The module to create an instruction for.", prefix="MODULE:",
            )
            module_description = dspy.InputField(
                desc="Description of the module to create an instruction for.", prefix="MODULE DESCRIPTION:",
            )
        task_demos = dspy.InputField(
            format=str,
            desc="Example inputs/outputs of our module.",
            prefix="TASK DEMO(S):",
        )
        if use_instruct_history:
            previous_instructions = dspy.InputField(
                format=str,
                desc="Previous instructions we've attempted, along with their associated scores.",
                prefix="PREVIOUS INSTRUCTIONS:",
            )
        basic_instruction = dspy.InputField(
            format=str, desc="Basic instruction.", prefix="BASIC INSTRUCTION:",
        )
        if use_tip:
            tip = dspy.InputField(
                format=str,
                desc="A suggestion for how to go about generating the new instruction.",
                prefix="TIP:",
            )

        # --- ADDED FIELD ---
        custom_knowledge_document = dspy.InputField(
            format=str,
            desc="A custom document providing relevant context, definitions, or examples (e.g., acronyms). Use this to inform the proposed instruction.",
            prefix="CUSTOM KNOWLEDGE DOCUMENT:",
        )
        # --- END ADDED FIELD ---

        proposed_instruction = dspy.OutputField(
            desc="Propose an instruction that will be used to prompt a Language Model to perform this task, incorporating relevant information from the custom knowledge document where appropriate.",
            prefix="PROPOSED INSTRUCTION:",
        )

    return dspy.Predict(GenerateSingleModuleInstructionWithKnowledge)


# --- Custom Instruction Generator Module ---
class CustomGenerateModuleInstruction(dspy.Module):
    def __init__(
        self,
        program_code_string=None,
        use_dataset_summary=True,
        program_aware=False,
        use_task_demos=True,
        use_instruct_history=True,
        use_tip=True,
        verbose=False,
    ):
        super().__init__()
        self.use_dataset_summary = use_dataset_summary
        self.program_aware = program_aware
        self.use_task_demos = use_task_demos
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip
        self.verbose = verbose

        self.program_code_string = program_code_string
        # These remain standard predictors
        self.describe_program = dspy.Predict(DescribeProgram)
        self.describe_module = dspy.Predict(DescribeModule)

        # --- USE CUSTOM SIGNATURE ---
        self.generate_module_instruction = generate_instruction_class_with_knowledge(
            use_dataset_summary=use_dataset_summary,
            program_aware=program_aware,
            use_task_demos=use_task_demos,
            use_instruct_history=use_instruct_history,
            use_tip=use_tip,
        )
        # --- END USE CUSTOM SIGNATURE ---

    # --- ADD custom_knowledge_document ARGUMENT ---
    def forward(
        self,
        demo_candidates,
        pred_i,
        demo_set_i,
        program,
        previous_instructions,
        data_summary,
        custom_knowledge_document, # Added argument
        num_demos_in_context=3,
        tip=None,
    ):
        # (Keep the original logic for gather_examples_from_sets helper function here)
        def gather_examples_from_sets(candidate_sets, max_examples):
            """Helper function to gather up to augmented examples from given sets."""
            count = 0
            for candidate_set in candidate_sets:
                for example in candidate_set:
                    if "augmented" in example.keys():
                        fields_to_use = get_signature(program.predictors()[pred_i]).fields
                        yield create_example_string(fields_to_use, example)
                        count += 1
                        if count >= max_examples:
                            return

        # (Keep the original logic for constructing task_demos here)
        basic_instruction = get_signature(program.predictors()[pred_i]).instructions
        task_demos = ""
        if self.use_task_demos and demo_candidates and demo_candidates[pred_i]: # Check if demo_candidates[pred_i] is not empty
            adjacent_sets = (
                [demo_candidates[pred_i][demo_set_i]] +
                demo_candidates[pred_i][demo_set_i + 1:] +
                demo_candidates[pred_i][:demo_set_i]
            )
            example_strings = gather_examples_from_sets(adjacent_sets, num_demos_in_context)
            task_demos = "\n\n".join(example_strings) + "\n\n"

        if not task_demos.strip() or demo_set_i == 0:
            task_demos = "No task demos provided."

        # (Keep the original logic for program_description, module_code, module_description here)
        program_description = "Not available"
        module_code = "Not provided"
        module_description = "Not provided"
        if self.program_aware:
            try:
                program_description = strip_prefix(
                    self.describe_program(
                        program_code=self.program_code_string, program_example=task_demos,
                    ).program_description,
                )
                if self.verbose:
                    print(f"PROGRAM DESCRIPTION: {program_description}")

                inputs = []
                outputs = []
                for field_name, field in get_signature(program.predictors()[pred_i]).fields.items():
                    dspy_field_type = field.json_schema_extra.get('__dspy_field_type')
                    if dspy_field_type == "input":
                        inputs.append(field_name)
                    else:
                        outputs.append(field_name)

                module_code = f"{program.predictors()[pred_i].__class__.__name__}({', '.join(inputs)}) -> {', '.join(outputs)}"

                module_description = self.describe_module(
                    program_code=self.program_code_string,
                    program_description=program_description,
                    program_example=task_demos,
                    module=module_code,
                ).module_description # Removed max_depth, not in original signature
            except Exception as e:
                 if self.verbose:
                    print(f"Error getting program description: {e}. Running without program aware proposer.")
                 self.program_aware = False # Corrected variable name


        if self.verbose:
            print(f"task_demos {task_demos}")

        # --- PASS custom_knowledge_document to the Predictor ---
        instruct = self.generate_module_instruction(
            dataset_description=data_summary,
            program_code=self.program_code_string,
            module=module_code,
            program_description=program_description,
            module_description=module_description,
            task_demos=task_demos,
            tip=tip,
            basic_instruction=basic_instruction,
            previous_instructions=previous_instructions,
            custom_knowledge_document=custom_knowledge_document, # Pass the document here
        )
        # --- END PASS ---

        proposed_instruction = strip_prefix(instruct.proposed_instruction)

        return dspy.Prediction(proposed_instruction=proposed_instruction)
    

class CustomKnowledgeProposer(GroundedProposer):
    # --- ADD custom_knowledge_document to __init__ ---
    def __init__(
        self,
        prompt_model,
        program,
        trainset,
        custom_knowledge_document, # Added argument
        view_data_batch_size=10,
        use_dataset_summary=True,
        program_aware=True,
        use_task_demos=True,
        num_demos_in_context=3,
        use_instruct_history=True,
        use_tip=True,
        set_tip_randomly=True,
        set_history_randomly=True,
        verbose=False,
        rng=None,
    ):
        # Call super().__init__ *without* the custom argument first
        super().__init__(
            prompt_model=prompt_model,
            program=program,
            trainset=trainset,
            view_data_batch_size=view_data_batch_size,
            use_dataset_summary=use_dataset_summary,
            program_aware=program_aware,
            use_task_demos=use_task_demos,
            num_demos_in_context=num_demos_in_context,
            use_instruct_history=use_instruct_history,
            use_tip=use_tip,
            set_tip_randomly=set_tip_randomly,
            set_history_randomly=set_history_randomly,
            verbose=verbose,
            rng=rng,
        )
        # Store the custom document
        self.custom_knowledge_document = custom_knowledge_document
    # --- END __init__ MODIFICATION ---

    # --- OVERRIDE propose_instruction_for_predictor ---
    def propose_instruction_for_predictor(
        self,
        program,
        predictor,
        pred_i,
        T,
        demo_candidates,
        demo_set_i,
        trial_logs,
        tip=None,
    ) -> str:
        """This method is responsible for returning a single instruction for a given predictor, using the specified criteria."""

        instruction_history = create_predictor_level_history_string(
            program, pred_i, trial_logs, MAX_INSTRUCT_IN_HISTORY,
        ) if self.use_instruct_history else "No history provided."

        # --- USE CUSTOM INSTRUCTION GENERATOR ---
        instruction_generator = CustomGenerateModuleInstruction(
            program_code_string=self.program_code_string,
            use_dataset_summary=self.use_dataset_summary,
            program_aware=self.program_aware,
            use_task_demos=self.use_task_demos and demo_candidates,
            use_instruct_history=self.use_instruct_history and instruction_history != "No history provided.", # Check if history is actually present
            use_tip=self.use_tip,
            verbose=self.verbose,
        )
        # --- END USE CUSTOM INSTRUCTION GENERATOR ---

        original_temp = self.prompt_model.kwargs.get("temperature", 1.0) # Use .get with default

        epsilon = self.rng.uniform(0.01, 0.05)
        modified_temp = T + epsilon

        with dspy.settings.context(lm=self.prompt_model):
            self.prompt_model.kwargs["temperature"] = modified_temp
            # --- PASS custom_knowledge_document to forward ---
            proposed_instruction = instruction_generator.forward(
                demo_candidates=demo_candidates,
                pred_i=pred_i,
                demo_set_i=demo_set_i,
                program=program,
                data_summary=self.data_summary,
                previous_instructions=instruction_history,
                custom_knowledge_document=self.custom_knowledge_document, # Pass stored document
                num_demos_in_context=self.num_demos_in_context,
                tip=tip,
            ).proposed_instruction
            # --- END PASS ---
        self.prompt_model.kwargs["temperature"] = original_temp

        if self.verbose:
            self.prompt_model.inspect_history(n=1)
            print(f"PROPOSED INSTRUCTION: {proposed_instruction}")

        return strip_prefix(proposed_instruction)


class MIPROv2WithCustomProposer(MIPROv2):
    # --- ADD custom_knowledge_document to __init__ ---
    def __init__(
        self,
        metric: Callable,
        custom_knowledge_document: str, # Added argument
        prompt_model: Optional[Any] = None,
        task_model: Optional[Any] = None,
        teacher_settings: Dict = {},
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        auto: Optional[Literal["light", "medium", "heavy"]] = "light",
        num_candidates: Optional[int] = None,
        num_threads: Optional[int] = None,
        max_errors: int = 10,
        seed: int = 9,
        init_temperature: float = 0.5,
        verbose: bool = False,
        track_stats: bool = True,
        log_dir: Optional[str] = None,
        metric_threshold: Optional[float] = None,
    ):
         # Call super init first
        super().__init__(
            metric=metric,
            prompt_model=prompt_model,
            task_model=task_model,
            teacher_settings=teacher_settings,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            auto=auto,
            num_candidates=num_candidates,
            num_threads=num_threads,
            max_errors=max_errors,
            seed=seed,
            init_temperature=init_temperature,
            verbose=verbose,
            track_stats=track_stats,
            log_dir=log_dir,
            metric_threshold=metric_threshold,
        )
        # Store the custom document
        self.custom_knowledge_document = custom_knowledge_document

        # --- ADDED Fallback Initialization ---
        # Ensure these attributes exist *after* super().__init__ has run.
        # The values might be None initially if num_candidates wasn't provided,
        # but they *should* be properly set later by _set_hyperparams_from_run_mode
        # based on the 'auto' setting. This just prevents the AttributeError
        # if something goes wrong before that recalculation.
        if not hasattr(self, 'num_fewshot_candidates'):
            self.num_fewshot_candidates = num_candidates
        if not hasattr(self, 'num_instruct_candidates'):
            if num_candidates is None:
                self.num_instruct_candidates = 3  # default to 3 if num_candidates is None
            else:
                self.num_instruct_candidates = max(num_candidates, 3)

    # --- END __init__ MODIFICATION ---

    # --- OVERRIDE _propose_instructions ---
    def _propose_instructions(
        self,
        program: Any,
        trainset: List,
        demo_candidates: Optional[List],
        view_data_batch_size: int,
        program_aware_proposer: bool,
        data_aware_proposer: bool,
        tip_aware_proposer: bool,
        fewshot_aware_proposer: bool,
    ) -> Dict[int, List[str]]:
        logger.info("\n==> STEP 2: PROPOSE INSTRUCTION CANDIDATES (Using Custom Proposer) <==")
        logger.info(
            "We will use the few-shot examples, dataset summary, program code, a prompting tip, and custom knowledge to propose instructions."
        )

        # --- INSTANTIATE CUSTOM PROPOSER ---
        proposer = CustomKnowledgeProposer(
            program=program,
            trainset=trainset,
            prompt_model=self.prompt_model,
            custom_knowledge_document=self.custom_knowledge_document + "\n\nNote you will not have access to this custom knowledge document when deployed; you must copy relevant portions to your proposed_instruction.", # append instructions
            view_data_batch_size=view_data_batch_size,
            program_aware=program_aware_proposer,
            use_dataset_summary=data_aware_proposer,
            use_task_demos=fewshot_aware_proposer,
            num_demos_in_context=BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT,
            use_tip=tip_aware_proposer,
            set_tip_randomly=tip_aware_proposer, # Assuming same control logic
            use_instruct_history=False, # Default from MIPROv2 source
            set_history_randomly=False, # Default from MIPROv2 source
            verbose=self.verbose,
            rng=self.rng,
        )
        # --- END INSTANTIATE CUSTOM PROPOSER ---

        logger.info(f"\nProposing N={self.num_instruct_candidates} instructions...\n")
        # The rest of the method remains the same as it uses the proposer instance
        instruction_candidates = proposer.propose_instructions_for_program(
            trainset=trainset, # Pass trainset explicitly
            program=program,
            demo_candidates=demo_candidates,
            N=self.num_instruct_candidates,
            T=self.init_temperature,
            trial_logs={}, # Pass trial_logs explicitly
        )

        for i, pred in enumerate(program.predictors()):
            logger.info(f"Proposed Instructions for Predictor {i}:\n")
            # Ensure the original instruction is the first candidate
            original_instruction = get_signature(pred).instructions
            if original_instruction not in instruction_candidates.get(i, []):
                 instruction_candidates.setdefault(i, []).insert(0, original_instruction)
            elif instruction_candidates[i][0] != original_instruction:
                instruction_candidates[i].remove(original_instruction)
                instruction_candidates[i].insert(0, original_instruction)

            for j, instruction in enumerate(instruction_candidates[i]):
                logger.info(f"{j}: {instruction}\n")
            logger.info("\n")

        return instruction_candidates
    