import json, random, pathlib, os
from dotenv import load_dotenv
from pydantic import BaseModel
import logging

import openai
from openai import OpenAI
from rouge_score import rouge_scorer
import dspy
#from dspy.teleprompt import MIPROv2
from custom_mipro import (
    MIPROv2WithCustomProposer,
    CustomKnowledgeProposer,
    CustomGenerateModuleInstruction,
    generate_instruction_class_with_knowledge
)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set root logger to DEBUG to allow all levels
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),  # File handler for DEBUG logs
        logging.StreamHandler()  # Console handler
    ]
)

# Set individual handler levels
logging.getLogger().handlers[0].setLevel(logging.DEBUG)  # File handler
logging.getLogger().handlers[0].setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s'))

logging.getLogger().handlers[1].setLevel(logging.INFO)   # Console handler

# load openai api key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


class JudgeScore(BaseModel):
    score: float

def llm_metric(example, pred, *_):
    prompt = f"""You are the QA oracle. Given:
    • user_request: {example.request}
    • model_answer: {pred.answer}
    • reference_answer: {example.answer}
    
    Return a JSON object with a single float 'score' between 0-10 measuring how well the model_answer satisfies the user compared to the reference_answer / how well the model_answer incorporates the helpful aspects of the reference_answer.
    """
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format=JudgeScore
    )
    return response.choices[0].message.parsed.score / 10

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def rouge_metric(example, pred, trace=None):
    score = scorer.score(example.answer, pred.answer)['rougeL'].fmeasure
    return score 

def combo_metric(example, pred, trace=None, alpha=.3):
    return alpha*rouge_metric(example, pred) + (1-alpha)*llm_metric(example, pred)


lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)
DATA_DIR = pathlib.Path("./data")
random.seed(347)

# ---------- load + join ----------
def create_examples(use_initial_query=True):
    out = []
    fb   = json.loads((DATA_DIR/"feedback_20250504_170617.json").read_text())
    pos  = {k for k,v in fb.items() if v["feedback"]=="positive"}

    for line in (DATA_DIR/"conversations.jsonl").read_text().splitlines():
        convo = json.loads(line)
        cid   = convo["id"]
        msgs  = convo["messages"]
        init  = next(m["content"] for m in msgs if m["role"]=="user")

        for idx, m in enumerate(msgs):
            if f"{cid}_{idx}" in pos and m["role"]=="assistant":
                inp = (init if use_initial_query
                       else next((x["content"] for x in reversed(msgs[:idx])
                                  if x["role"]=="user"), init))
                ex = dspy.Example(request=inp, answer=m["content"])\
                        .with_inputs("request")
                out.append(ex)
    return out

# Create examples
examples = create_examples(use_initial_query=True)

# ---------- split ----------
random.shuffle(examples)
val, train = examples[:int(.2*len(examples))], examples[int(.2*len(examples)):]

# ---------- signature + module ----------
class ConvQA(dspy.Signature):
    """You are a helpful assistant at QuantGlyph Systems, powered by a large language model."""
    request: str = dspy.InputField()
    answer:  str = dspy.OutputField()

qa = dspy.Predict(ConvQA)

tp = MIPROv2WithCustomProposer(metric=combo_metric,
                               custom_knowledge_document=open("generate_data/quantglyph_org.md", 'r').read(),
                               #prompt_model=dspy.LM('openai/o4-mini', api_key=os.getenv("OPENAI_API_KEY"), max_tokens=20000, temperature=1.0),
                               prompt_model=dspy.LM('gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY")),
                               auto="medium")

best = tp.compile(qa, trainset=train, valset=val)
best.save("basic_prompt.json")

# output the final system prompt
print()
print("Final system prompt:")
print(best.signature.instructions)
print()
