import json, random, pathlib, os
from dotenv import load_dotenv
from pydantic import BaseModel

import openai
from openai import OpenAI
from rouge_score import rouge_scorer
import dspy
from dspy.teleprompt import MIPROv2


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
    fb   = json.loads((DATA_DIR/"feedback_20250504_123148.json").read_text())
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
    """You are a helpful assistant powered by a large language model."""
    request: str            = dspy.InputField()
    answer:  str            = dspy.OutputField()

qa = dspy.Predict(ConvQA)

tp = MIPROv2(metric=combo_metric,
            #teacher_settings={"extra_content": open("generate_data/quantglyph_org.md", "r").read()},
            auto="medium")

best = tp.compile(qa, trainset=train, valset=val)
best.save("basic_prompt.json")

# output the final system prompt
print()
print("Final system prompt:")
print(best.signature.instructions)
print()
