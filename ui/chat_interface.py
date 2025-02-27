"""
Chat interface components for the Streamlit UI.
"""
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

def initialize_chat_state():
    """Initialize chat state in the Streamlit session."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "thread_id" not in st.session_state:
        # Generate a unique thread ID using timestamp
        st.session_state.thread_id = f"thread_{int(datetime.now().timestamp())}"


def display_chat_messages():
    """Display chat messages in the Streamlit UI."""
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        
        with st.chat_message(role):
            st.markdown(content)


def handle_user_input(on_submit_callback: Callable[[str, str], List[Dict]]):
    """
    Handle user input in the chat interface.
    
    Args:
        on_submit_callback: Callback function that takes a message and thread_id
                          and returns a list of response messages
    """
    if prompt := st.chat_input("What would you like help with?"):
        # Add user message to chat history
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        # Display the user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from the agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                responses = on_submit_callback(prompt, st.session_state.thread_id)
                
                # Process and display responses
                for response in responses:
                    st.markdown(response.get("content", ""))
                    # Add to chat history
                    st.session_state.messages.append(response)