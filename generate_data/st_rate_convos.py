import streamlit as st
import jsonlines
import json
from datetime import datetime
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Conversation Viewer",
    page_icon="💬",
    layout="wide"
)

# Initialize session state for storing feedback
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

def load_conversations(file_path):
    conversations = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            conversations.append(obj)
    return conversations

def save_feedback(feedback_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"feedback_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    return output_file

# Main app
st.title("💬 Conversation Viewer")

# Load conversations
conversations = load_conversations("conversations.jsonl")

# Sidebar with conversation selection
st.sidebar.title("Navigation")
conversation_index = st.sidebar.selectbox(
    "Select Conversation",
    range(len(conversations)),
    format_func=lambda x: f"Conversation {x + 1}"
)

# Export feedback button
if st.sidebar.button("Export Feedback"):
    if st.session_state.feedback:
        output_file = save_feedback(st.session_state.feedback)
        st.sidebar.success(f"Feedback exported to {output_file}")
    else:
        st.sidebar.warning("No feedback to export")

# Display selected conversation
if conversation_index is not None:
    conversation = conversations[conversation_index]
    
    # Create columns for the main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"Conversation {conversation_index + 1}")
        
        # Display messages
        for msg_idx, message in enumerate(conversation.get("messages", [])):
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Style based on role
            if role == "user":
                st.markdown("**User:**")
                st.markdown(content)
            elif role == "assistant":
                msg_id = f"{conversation_index}_{msg_idx}"
                
                # Assistant message with feedback buttons
                st.markdown("**Assistant:**")
                st.markdown(content)
                
                # Feedback buttons
                col_up, col_down = st.columns([1, 20])
                with col_up:
                    if st.button("👍", key=f"up_{msg_id}"):
                        st.session_state.feedback[msg_id] = {
                            "conversation_id": conversation_index,
                            "message_index": msg_idx,
                            "feedback": "positive",
                            "content": content
                        }
                        st.success("Positive feedback recorded!")
                with col_down:
                    if st.button("👎", key=f"down_{msg_id}"):
                        st.session_state.feedback[msg_id] = {
                            "conversation_id": conversation_index,
                            "message_index": msg_idx,
                            "feedback": "negative",
                            "content": content
                        }
                        st.error("Negative feedback recorded!")
            
            st.markdown("---")
    
    # Show feedback status in the sidebar
    with col2:
        st.subheader("Feedback Status")
        feedback_count = len(st.session_state.feedback)
        st.metric("Total Feedback Given", feedback_count)
        
        # Show feedback for current conversation
        current_feedback = {k: v for k, v in st.session_state.feedback.items() 
                          if v["conversation_id"] == conversation_index}
        if current_feedback:
            st.markdown("### Current Conversation Feedback")
            for msg_id, feedback in current_feedback.items():
                emoji = "👍" if feedback["feedback"] == "positive" else "👎"
                st.markdown(f"Message {feedback['message_index']}: {emoji}") 