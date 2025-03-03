"""
Main Streamlit page for the onboarding agent with document upload functionality.
"""
import logging
import os
from datetime import datetime
from typing import Optional

import streamlit as st

from agents.onboarding_agent import OnboardingAgent
from retrivers.code_retriever import CodeRetriever
from retrivers.document_retriever import DocumentRetriever
from retrivers.loaders.pdf_loader import PDFDocumentLoader
from ui.chat_interface import display_chat_messages, handle_user_input, initialize_chat_state
from ui.components import render_knowledge_base_stats, render_onboarding_progress, render_sidebar

logger = logging.getLogger(__name__)

def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Genie Assistant",
        page_icon="🧞‍♂️",
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
    
    /* Fix for chat input at bottom */
    .main .block-container {
        padding-bottom: 5rem;
    }
    
    /* Upload file container */
    .upload-container {
        border: 1px dashed #aaa;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
        background-color: #f8f9fa;
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

def handle_document_upload():
    """Handle document upload and processing."""
    with st.expander("Upload Documents", expanded=False):
        st.write("Upload documents to ask questions about them")
        
        # Create upload container
        uploaded_files = st.file_uploader(
            "Choose files (PDF, TXT, etc.)", 
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "csv", "json", "py", "js", "html", "css", ".go"]
        )
        
        # Process uploaded files
        if uploaded_files:
            if st.button("Process Uploaded Documents"):
                with st.spinner("Processing documents..."):
                    processed_files = []
                    
                    # Create temp directory to save files
                    os.makedirs("temp_uploads", exist_ok=True)
                    
                    # Process each file
                    for uploaded_file in uploaded_files:
                        try:
                            # Save file temporarily
                            file_path = os.path.join("temp_uploads", uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Check file type and process accordingly
                            if uploaded_file.name.endswith(".pdf"):
                                # Process PDF
                                pdf_loader = PDFDocumentLoader()
                                documents = pdf_loader.load_document(file_path)
                                
                                # Add to knowledge base
                                doc_retriever = DocumentRetriever()
                                doc_retriever.knowledge_base.add_documents(documents, source_type="pdf")
                                
                                processed_files.append(f"PDF: {uploaded_file.name} ({len(documents)} pages)")
                            
                            elif uploaded_file.name.endswith((".txt", ".md")):
                                # Process text file as a single document
                                from langchain_core.documents import Document
                                
                                with open(file_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                
                                document = Document(
                                    page_content=content,
                                    metadata={
                                        "source": uploaded_file.name,
                                        "source_type": "text",
                                    }
                                )
                                
                                # Add to knowledge base
                                doc_retriever = DocumentRetriever()
                                doc_retriever.knowledge_base.add_documents([document], source_type="text")
                                
                                processed_files.append(f"Text: {uploaded_file.name}")
                            
                            elif uploaded_file.name.endswith((".py", ".js", ".html", ".css", ".go")):
                                # Process code file
                                from langchain_core.documents import Document
                                
                                with open(file_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                
                                document = Document(
                                    page_content=content,
                                    metadata={
                                        "source": uploaded_file.name,
                                        "source_type": "code",
                                        "language": uploaded_file.name.split(".")[-1],
                                    }
                                )
                                
                                # Add to knowledge base
                                code_retriever = CodeRetriever()
                                code_retriever.knowledge_base.add_documents([document], source_type="code")
                                
                                processed_files.append(f"Code: {uploaded_file.name}")
                            
                            else:
                                st.warning(f"Unsupported file type: {uploaded_file.name}")
                        
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    
                    # Display results
                    if processed_files:
                        st.success(f"Successfully processed {len(processed_files)} files")
                        for file_info in processed_files:
                            st.write(f"✅ {file_info}")
                        
                        # Add a message to the chat
                        if "messages" in st.session_state:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"I've processed {len(processed_files)} documents. You can now ask questions about them!"
                            })

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
        st.title("🧞‍♂️ Genie Assistant")
        st.caption("Ask questions about the codebase, documentation, or get guidance on onboarding steps")
        
        # Document upload section
        handle_document_upload()
        
        # Knowledge base stats
        render_knowledge_base_stats(doc_stats, code_stats)
        
        # Onboarding progress (if available)
        if onboarding_progress:
            render_onboarding_progress(onboarding_progress)
        
        # Separator
        st.markdown("---")
        
        # Chat container - using columns to push chat to bottom
        chat_container = st.container()
        
        with chat_container:
            # Chat interface
            st.subheader("💬 Chat")
            
            # Display chat messages
            display_chat_messages()
            
            # Add some space before the input box
            st.markdown("<div style='height: 50px'></div>", unsafe_allow_html=True)
            
            # Function to handle new messages
            def handle_message(message: str, thread_id: str):
                return agent.invoke(message, thread_id)
            
            # User input at the bottom
            handle_user_input(handle_message)