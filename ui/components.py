"""
Reusable UI components for the Streamlit app.
"""
from datetime import datetime
import logging
from typing import Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

def render_sidebar():
    """Render the sidebar with controls and information."""
    with st.sidebar:
        st.title("Onboarding Assistant 🧞‍♂️")
        st.subheader("Features")
        
        # Features list
        st.markdown("""
        - 📚 Documentation Search
        - 💻 Code Exploration
        - 🚀 Guided Onboarding
        - 🧠 Context-Aware Responses
        - 👤 Human Expert Escalation
        """)
        
        # Thread ID info
        if "thread_id" in st.session_state:
            st.subheader("Session Info")
            st.code(f"Thread ID: {st.session_state.thread_id}")
        
        # Reset conversation button
        if st.button("Start New Conversation"):
            st.session_state.messages = []
            st.session_state.thread_id = f"thread_{int(datetime.now().timestamp())}"
            st.rerun()
        
        # Add footer
        st.markdown("---")
        st.caption("Powered by LangGraph, LangChain, and Azure OpenAI")


def render_knowledge_base_stats(doc_stats: Optional[Dict] = None, code_stats: Optional[Dict] = None):
    """
    Render knowledge base statistics.
    
    Args:
        doc_stats: Document knowledge base statistics
        code_stats: Code knowledge base statistics
    """
    with st.expander("Knowledge Base Stats", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Documentation")
            if doc_stats:
                st.metric("Documents", doc_stats.get("document_count", 0))
                st.text(f"Collection: {doc_stats.get('collection_name', 'N/A')}")
            else:
                st.info("No document statistics available")
        
        with col2:
            st.subheader("Code Repository")
            if code_stats:
                st.metric("Files", code_stats.get("document_count", 0))
                st.text(f"Collection: {code_stats.get('collection_name', 'N/A')}")
            else:
                st.info("No code statistics available")


def render_onboarding_progress(progress_data: Optional[Dict] = None):
    """
    Render onboarding progress information.
    
    Args:
        progress_data: Onboarding progress data
    """
    if not progress_data:
        return
    
    with st.expander("Onboarding Progress", expanded=True):
        # User info
        st.subheader(f"{progress_data.get('user_name', 'Team Member')}")
        st.caption(f"Role: {progress_data.get('role', 'N/A')}")
        
        # Progress bar
        completed = len(progress_data.get("completed_steps", []))
        remaining = len(progress_data.get("remaining_steps", []))
        total = completed + remaining
        
        if total > 0:
            progress_percentage = completed / total
            st.progress(progress_percentage)
            st.caption(f"{completed} of {total} steps completed ({int(progress_percentage * 100)}%)")
        
        # Steps
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Completed Steps")
            for step in progress_data.get("completed_steps", []):
                st.success(step)
        
        with col2:
            st.subheader("Remaining Steps")
            current_step = progress_data.get("current_step")
            
            for step in progress_data.get("remaining_steps", []):
                if step == current_step:
                    st.info(f"📍 {step} (Current)")
                else:
                    st.text(step)