"""
Main Streamlit page for the onboarding agent.
"""
import logging
from datetime import datetime
from typing import Optional

import streamlit as st

from agents.onboarding_agent import OnboardingAgent
from retrieval.code_retriever import CodeRetriever
from retrieval.document_retriever import DocumentRetriever
from ui.chat_interface import display_chat_messages, handle_user_input, initialize_chat_state
from ui.components import render_knowledge_base_stats, render_onboarding_progress, render_sidebar

logger = logging.getLogger(__name__)

def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Team Onboarding Assistant",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Set custom CSS
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    .stChatMessage.user {
        background-color: #e6f7ff;
    }
    
    .stChatMessage.assistant {
        background-color: #f0f7ea;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_agent() -> OnboardingAgent:
    """
    Initialize the onboarding agent.
    
    Returns:
        Initialized OnboardingAgent instance
    """
    if "agent" not in st.session_state:
        with st.spinner("Initializing AI assistant..."):
            st.session_state.agent = OnboardingAgent()
    
    return st.session_state.agent


def get_knowledge_base_stats():
    """
    Get statistics about the knowledge bases.
    
    Returns:
        Tuple of (document_stats, code_stats)
    """
    try:
        doc_retriever = DocumentRetriever()
        code_retriever = CodeRetriever()
        
        return doc_retriever.get_stats(), code_retriever.get_stats()
    except Exception as e:
        logger.error(f"Error getting knowledge base stats: {e}")
        return None, None


def get_onboarding_progress(thread_id: str, agent: OnboardingAgent) -> Optional[dict]:
    """
    Get onboarding progress for the current thread.
    
    Args:
        thread_id: Thread ID
        agent: Onboarding agent instance
    
    Returns:
        Onboarding progress data or None
    """
    try:
        # Get the thread history
        history = agent.get_thread_history(thread_id)
        
        # Try to get the current graph state
        config = {"configurable": {"thread_id": thread_id}}
        current_state = agent.graph.get_state(config)
        
        if current_state and "onboarding_progress" in current_state.values:
            return current_state.values["onboarding_progress"]
        
        return None
    
    except Exception as e:
        logger.error(f"Error getting onboarding progress: {e}")
        return None


def render_main_page():
    """Render the main Streamlit page."""
    setup_page()
    
    # Initialize chat state
    initialize_chat_state()
    
    # Initialize agent
    agent = initialize_agent()
    
    # Get knowledge base stats
    doc_stats, code_stats = get_knowledge_base_stats()
    
    # Get onboarding progress
    onboarding_progress = get_onboarding_progress(st.session_state.thread_id, agent)
    
    # Render sidebar
    render_sidebar()
    
    # Main container
    main_container = st.container()
    
    with main_container:
        # Header
        st.title("🚀 Team Onboarding Assistant")
        st.caption("Ask questions about the codebase, documentation, or get guidance on onboarding steps")
        
        # Knowledge base stats
        render_knowledge_base_stats(doc_stats, code_stats)
        
        # Onboarding progress (if available)
        if onboarding_progress:
            render_onboarding_progress(onboarding_progress)
        
        # Separator
        st.markdown("---")
        
        # Chat interface
        st.subheader("💬 Chat")
        display_chat_messages()
        
        # Function to handle new messages
        def handle_message(message: str, thread_id: str):
            return agent.invoke(message, thread_id)
        
        # User input
        handle_user_input(handle_message)