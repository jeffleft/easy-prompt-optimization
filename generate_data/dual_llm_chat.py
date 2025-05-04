#!/usr/bin/env python3
"""
dual_llm_chat.py

Simulate a multi-turn conversation between two LLMs using the OpenAI ChatCompletion API.
One LLM acts as a user with a defined persona and goal, and the other LLM acts as an assistant with a custom system prompt.
"""

import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import json
import random


load_dotenv()
client = OpenAI()


def evaluate_conversation_completion(messages: list, judge_model: str = "gpt-4") -> bool:
    """Use an LLM to evaluate if the conversation has reached a natural conclusion."""
    # Format the conversation for the judge
    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in messages
    ])
    
    judge_prompt = f"""You are an expert at evaluating conversations between a user and an AI assistant. 
Your task is to determine if the conversation has reached a natural conclusion.

A conversation should be considered complete if either:
1. The user has gotten what they wanted (the user must explicitly say so, which means you MUST NOT judge the conversation as complete if the user hasn't responded to the initial assistant response!)
2. It's clear the user will not be able to accomplish their goal
3. The conversation has reached a dead end or stalemate (e.g. the user is repeating themselves, or the assistant is repeating the same response)

Here is the conversation to evaluate:

{conversation_text}

Evaluate if the conversation is complete. Respond with ONLY "COMPLETE" or "CONTINUE"."""

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )
    
    decision = response.choices[0].message.content.strip().upper()
    print(f"Judge prompt: {judge_prompt}")
    print(f"Judge decision: {decision}")
    return decision == "COMPLETE"

def simulate_conversation(user_system_prompt: str,
                          assistant_system_prompt: str,
                          user_model: str,
                          assistant_model: str,
                          n_exchanges: int,
                          use_judge: bool = False,
                          judge_model: str = "gpt-4.1",
                          max_exchanges: int = 10) -> list:
    """Run back-and-forth exchanges between the user and assistant LLM. Returns a list of messages with IDs."""
    messages = []
    user_message_history = []
    assistant_message_history = []
    exchange_count = 0

    while True:
        # Check if we've hit the maximum exchanges
        if exchange_count >= max_exchanges:
            break
            
        # User turn
        if not user_message_history:
            user_message_history = [
                {"role": "system", "content": user_system_prompt},
                {"role": "user", "content": "Generate the initial message to the assistant."}
            ]
            assistant_message_history = [
                {"role": "system", "content": assistant_system_prompt}
            ]

        response = client.chat.completions.create(
            model=user_model,
            messages=user_message_history
        )
        print("User message: ", user_message_history)
        user_message = response.choices[0].message.content.strip()
        user_message_history.append({"role": "assistant", "content": user_message})
        assistant_message_history.append({"role": "user", "content": user_message})

        messages.append({"id": len(messages)+1, "role": "user", "content": user_message})
        
        # Check if we should continue based on judge or fixed exchanges
        # Only check after the first user turn (when we have at least 2 messages)
        if use_judge and len(messages) >= 2 and evaluate_conversation_completion(messages, judge_model):
            break
        elif not use_judge and exchange_count >= n_exchanges:
            break
        
        # Assistant turn
        response = client.chat.completions.create(
            model=assistant_model,
            messages=assistant_message_history
        )
        print("Assistant message: ", assistant_message_history)
        assistant_message = response.choices[0].message.content.strip()
        assistant_message_history.append({"role": "assistant", "content": assistant_message})
        user_message_history.append({"role": "user", "content": "Assistant response: "+assistant_message})

        messages.append({"id": len(messages)+1, "role": "assistant", "content": assistant_message})
        
        exchange_count += 1

    return messages


def main():
    parser = argparse.ArgumentParser(
        description="Simulate conversations between LLMs using prompt configurations from a JSON file."
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to JSON file containing an array of configs with 'user_system_prompt' and 'assistant_system_prompt'."
    )
    parser.add_argument(
        "--output_file", type=str, default="conversations.jsonl",
        help="Path to output JSONL file where conversations will be written."
    )
    parser.add_argument(
        "--user_model", type=str, default="gpt-4.1",
        help="Default model name to use for the user LLM."
    )
    parser.add_argument(
        "--assistant_model", type=str, default="gpt-4.1",
        help="Default model name to use for the assistant LLM."
    )
    parser.add_argument(
        "--n_exchanges", type=int, default=5,
        help="Default number of exchanges per conversation."
    )
    parser.add_argument(
        "--use_judge", action="store_true",
        help="Use an LLM judge to determine when the conversation should end instead of fixed number of exchanges."
    )
    parser.add_argument(
        "--judge_model", type=str, default="gpt-4.1",
        help="Model to use as the conversation judge."
    )
    parser.add_argument(
        "--max_exchanges", type=int, default=10,
        help="Maximum number of exchanges allowed when using the judge."
    )
    parser.add_argument(
        "--test", type=bool, default=False,
        help="Test mode: only run 1 conversation - picked at random from the input file"
    )
    args = parser.parse_args()

    # Read configurations array from input JSON
    with open(args.input_file, 'r') as f:
        configs = json.load(f)
    
    if args.test:
        # Pick a random config from the input file
        random_config = random.choice(configs)
        print(f"Running test conversation with config: {random_config}")
        messages = simulate_conversation(
            random_config["user_system_prompt"],
            random_config["assistant_system_prompt"],
            random_config.get("user_model", args.user_model),
            random_config.get("assistant_model", args.assistant_model),
            random_config.get("n_exchanges", args.n_exchanges),
            args.use_judge,
            args.judge_model,
            args.max_exchanges
        )
        with open(args.output_file, 'w') as out_f:
            conversation = {"id": 1, "messages": messages}  
            out_f.write(json.dumps(conversation) + "\n")
        print(f"Finished writing test conversation to {args.output_file}.")
        return

    # Write each conversation as a JSON line
    with open(args.output_file, 'w') as out_f:
        for cid, cfg in enumerate(configs, 1):
            user_prompt = cfg.get("user_system_prompt")
            assistant_prompt = cfg.get("assistant_system_prompt")
            if not user_prompt or not assistant_prompt:
                print(f"Skipping config {cid}, missing prompts.")
                continue
            # Allow per-config overrides
            user_model = cfg.get("user_model", args.user_model)
            assistant_model = cfg.get("assistant_model", args.assistant_model)
            n_exchanges = cfg.get("n_exchanges", args.n_exchanges)

            messages = simulate_conversation(
                user_prompt,
                assistant_prompt,
                user_model,
                assistant_model,
                n_exchanges,
                args.use_judge,
                args.judge_model,
                args.max_exchanges
            )
            conversation = {"id": cid, "messages": messages}
            out_f.write(json.dumps(conversation) + "\n")
    print(f"Finished writing {len(configs)} conversations to {args.output_file}.")


if __name__ == "__main__":
    main() 