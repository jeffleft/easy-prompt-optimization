import streamlit as st
import json
import pandas as pd
import plotly.express as px
from pathlib import Path
import openai
from typing import List, Dict, Any, Set
import time
import os
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Chat Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Constants
CONVERSATIONS_FILE = "data/conversations.jsonl"
FEEDBACK_FILE = "data/feedback_20250504_170617.json"

def load_conversations() -> List[Dict[str, Any]]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(CONVERSATIONS_FILE, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))
    return conversations

def load_feedback() -> Dict[str, str]:
    """Load feedback data from JSON file."""
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def get_basic_metrics(conversations: List[Dict[str, Any]], feedback: Dict[str, str]) -> Dict[str, Any]:
    """Calculate basic metrics without GPT-4 analysis."""
    total_conversations = len(conversations)
    total_messages = sum(len(conv["messages"]) for conv in conversations)
    avg_messages = total_messages / total_conversations if total_conversations > 0 else 0
    
    # Count feedback
    positive_feedback = sum(1 for f in feedback.values() if f.get("feedback") == "positive")
    negative_feedback = sum(1 for f in feedback.values() if f.get("feedback") == "negative")
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "avg_messages": avg_messages,
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback
    }

def generate_task_types(conversations: List[Dict[str, Any]], num_samples: int) -> Set[str]:
    """Generate task types by sampling conversations and using GPT-4.1-nano."""
    # Sample conversations
    sampled_convs = random.sample(conversations, min(num_samples, len(conversations)))
    
    # Extract just first user message from each sampled conversation
    samples = []
    for conv in sampled_convs:
        first_user_msg = next((msg["content"] for msg in conv["messages"] 
                             if msg["role"] == "user"), "")
        if first_user_msg:
            samples.append(first_user_msg)
    
    # Prepare prompt for GPT
    prompt = f"""Based on these conversation starters, identify distinct task types. 
Return only a JSON array of task type strings, no explanation. The task types should reflect the type of work the user is trying to accomplish and not necessarily the output of the work (eg. drafting an email is a single type of work regardless of topic or audience). Crucially: tasks within the same type should take similar amounts of time. No more than 12 task types.

Sample conversations:
{json.dumps(samples, indent=2)}

Example response format:
["Task Type 1", "Task Type 2", "Task Type 3"]"""

    try:
        # Call GPT
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an expert at categorizing technical conversations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # Parse the response
        task_types = json.loads(response.choices[0].message.content)
        return set(task_types)  # Convert to set to remove duplicates
    except Exception as e:
        st.error(f"Error generating task types: {str(e)}")
        return {"Other"}  # Fallback to a single category

def classify_conversation(conversation: Dict[str, Any], task_types: Set[str]) -> Dict[str, Any]:
    """Use GPT to classify the conversation and estimate time savings."""
    # Extract conversation content
    messages = conversation["messages"]
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    # Prepare the prompt for GPT
    prompt = f"""Analyze this conversation and provide:
1. Task type (choose from: {', '.join(sorted(task_types))})
2. Estimated time saved in minutes (based on task complexity and number of messages)
3. Success indicator (True if the conversation reached a clear resolution)

Conversation:
{conversation_text}

Provide the response in JSON format:
{{
    "task_type": "string",
    "time_saved_minutes": number,
    "success": boolean
}}"""

    try:
        # Call GPT
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing technical conversations and estimating time savings."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.error(f"Error analyzing conversation: {str(e)}")
        return {
            "task_type": "Other",
            "time_saved_minutes": 0,
            "success": False
        }

def analyze_conversations(conversations: List[Dict[str, Any]], task_types: Set[str]) -> pd.DataFrame:
    """Analyze conversations and return a DataFrame with results."""
    results = []
    
    with st.spinner("Analyzing conversations..."):
        for conv in conversations:
            analysis = classify_conversation(conv, task_types)
            results.append({
                "conversation_id": conv["id"],
                "task_type": analysis["task_type"],
                "success": analysis["success"],
                "time_saved_minutes": analysis["time_saved_minutes"]
            })
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
    
    return pd.DataFrame(results)

def main():
    st.title("Chat Analysis Dashboard")
    
    # Load data
    conversations = load_conversations()
    feedback = load_feedback()
    
    # Calculate basic metrics
    metrics = get_basic_metrics(conversations, feedback)
    
    # Display basic metrics
    st.subheader("Basic Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Conversations", metrics["total_conversations"])
    with col2:
        st.metric("Total Messages", metrics["total_messages"])
    with col3:
        st.metric("Avg Messages/Conv", f"{metrics['avg_messages']:.1f}")
    with col4:
        st.metric("Positive Feedback", metrics["positive_feedback"])
    with col5:
        st.metric("Negative Feedback", metrics["negative_feedback"])
    
    # Add controls for task type generation
    st.subheader("Task Type Generation")
    num_samples = st.slider(
        "Number of conversations to sample for task types",
        min_value=1,
        max_value=min(100, len(conversations)),
        value=5,
        help="More samples may give better task type coverage but will use more API calls"
    )
    
    if st.button("Generate Task Types"):
        with st.spinner("Generating task types..."):
            task_types = generate_task_types(conversations, num_samples)
            st.session_state.task_types = task_types
            st.write("Generated task types:", sorted(task_types))
    
    # Add button for GPT-4.1-nano analysis
    if st.button("Run Analysis"):
        if not hasattr(st.session_state, 'task_types'):
            st.error("Please generate task types first!")
            return
            
        # Analyze conversations
        df = analyze_conversations(conversations, st.session_state.task_types)
        
        # Display analysis results
        st.subheader("Analysis Results")
        
        # Task type distribution
        st.subheader("Task Type Distribution")
        fig = px.pie(df, names='task_type', title='Conversations by Task Type')
        st.plotly_chart(fig)
        
        # Success rate by task type
        st.subheader("Success Rate by Task Type")
        success_by_type = df.groupby('task_type')['success'].mean().reset_index()
        fig = px.bar(success_by_type, x='task_type', y='success',
                    title='Success Rate by Task Type')
        st.plotly_chart(fig)
        
        # Time savings by task type
        st.subheader("Time Savings by Task Type")
        col1, col2 = st.columns(2)
        
        with col1:
            # Total time saved
            time_by_type = df.groupby('task_type')['time_saved_minutes'].sum().reset_index()
            fig = px.bar(time_by_type, x='task_type', y='time_saved_minutes',
                        title='Total Time Saved by Task Type')
            st.plotly_chart(fig)
            
        with col2:
            # Average time saved
            avg_time_by_type = df.groupby('task_type')['time_saved_minutes'].mean().reset_index()
            fig = px.bar(avg_time_by_type, x='task_type', y='time_saved_minutes',
                        title='Average Time Saved by Task Type')
            st.plotly_chart(fig)
        
        # Raw data
        st.subheader("Raw Data")
        st.dataframe(df)

if __name__ == "__main__":
    main()
