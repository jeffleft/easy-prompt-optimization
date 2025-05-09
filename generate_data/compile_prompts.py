import json
import re

ASSISTANT_PROMPT = "You are a helpful assistant at QuantGlyph Systems, powered by a large language model."
DEFAULT_N = 3

USER_PROMPT = """You are a $ROLE at QuantGlyph Systems$PRODUCT

Here is some background information on the company and its products:
$COMPANY_BACKGROUND

Today you have the following task: $TASK.

Note that the task above may be underspecified. You should invent additional (but consistent) details as needed, although it is also reasonable to not now certain details and communicate that fact to the assistant. If there are acroynms (eg. product names) in the content in the assistant's response, make sure to ask it to expand them (it won't know what they are likely, so give it the necessary context).

You have been encouraged to use QG's internal chat assistant, so you try to accomplish your task with it's help.
However, you aren't very good at using LLM-based tools. You might interact with it more like a co-worker or inputting keywords into google.
Be realistic in emulating how a human chats (esp. consider: it's very tedious to type more than a few words). DO NOT repeat yourself (keep your previous responses in mind!) If the assistant is not giving you want you need, and there's no new information to provide, end the conversation (you can express your frustration with the assistant, but don't repeat yourself).
You are myopic on the task at hand, and may forget to include necessary information/context until prompted by the internal chat assistant. Importantly, in your first message you must assume that the assistant already knows the background information provided above, and only fill in the context provided in the background section if necessary in the next messages.
If the internal chat assistant makes assumptions or adds information that wasn't specifically provided by you, you must call it out. Note the assistant doesn't have access to any company information (Jira ticket, slack messages, code, table schema, ANYTHING!) so it will make things up. That is not acceptable - this must come from you. Instruct it to not make things up only IF you see it doing so. If the assistant asks you to provide more information (eg. from slack, jira, database, background information, etc.), you should do so (feel free to generate data that aligns with the company background and task).
If you are done (you've gotten the information you need or feel like the assistant can't help you), you should end the conversation (eg. "okay, thanks" or "ugh, nvm").
Do not deviate from this persona.

Here is a sample exchange to give you an idea of what an interaction might look like:
$SAMPLE_EXCHANGE"""

def extract_product_from_task(task):
    """Extract the product code from the task description."""
    # Look for QG- followed by uppercase letters
    match = re.search(r'QG-([A-Z]+)', task)
    if match:
        return f", working on the QG-{match.group(1)} product."
    return "."

def format_sample_exchange(sample_convo):
    """Format the sample conversation into a readable string."""
    formatted = []
    for msg in sample_convo["messages"]:
        formatted.append(f"{msg['role']}: {msg['content']}")
    return "\n".join(formatted)

def main():
    # Load topics and sample conversation
    with open("topics.json", "r") as f:
        topics = json.load(f)
    
    with open("sample_convo.json", "r") as f:
        sample_convo = json.load(f)
    
    # Format sample exchange
    sample_exchange = format_sample_exchange(sample_convo)

    # Load company background
    with open("quantglyph_org.md", "r") as f:
        company_background = f.read()
    
    # Generate prompts for each role and task
    prompts = []
    for role, tasks in topics.items():
        for task in tasks:
            product = extract_product_from_task(task)
            user_prompt = USER_PROMPT.replace("$ROLE", role)
            user_prompt = user_prompt.replace("$PRODUCT", product)
            user_prompt = user_prompt.replace("$TASK", task)
            user_prompt = user_prompt.replace("$SAMPLE_EXCHANGE", sample_exchange)
            user_prompt = user_prompt.replace("$COMPANY_BACKGROUND", company_background)
            
            prompts.append({
                "user_system_prompt": user_prompt,
                "assistant_system_prompt": ASSISTANT_PROMPT,
                "n_exchanges": DEFAULT_N
            })
    
    # Write to prompts.json
    with open("prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)

if __name__ == "__main__":
    main()