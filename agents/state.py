"""
State definitions for the onboarding agent LangGraph workflow.
"""
from typing import Annotated, Dict, List, Optional, Union

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class OnboardingProgress(BaseModel):
    """Tracks a user's progress through the onboarding process."""
    
    # User info
    user_id: str
    user_name: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    
    # Progress tracking
    completed_steps: List[str] = Field(default_factory=list)
    remaining_steps: List[str] = Field(default_factory=list)
    current_step: Optional[str] = None
    
    # Preferences
    preferred_docs_format: Optional[str] = None
    areas_of_interest: List[str] = Field(default_factory=list)


class KnowledgeGap(BaseModel):
    """Represents a gap in the knowledge base detected by the system."""
    
    query: str
    timestamp: str
    category: str
    confidence_score: float
    suggested_resources: Optional[List[str]] = None


class RetrievedContext(BaseModel):
    """Represents context retrieved from various sources."""
    
    content: str
    source: str
    source_type: str  # e.g., "code", "document", "pdf"
    relevance_score: float
    metadata: Dict = Field(default_factory=dict)


class State(TypedDict):
    """The state maintained by the LangGraph workflow."""
    
    # Conversation tracking - using LangGraph's add_messages reducer
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Query classification and routing
    query_type: Optional[str]
    confidence: Optional[float]
    
    # Retrieved information
    retrieved_contexts: List[RetrievedContext]
    
    # Onboarding state
    onboarding_progress: Optional[OnboardingProgress]
    
    # Knowledge gaps tracking
    knowledge_gaps: List[KnowledgeGap]
    
    # Human escalation tracking
    escalated: bool
    escalation_reason: Optional[str]
    human_response: Optional[str]


def create_initial_state() -> State:
    """
    Create the initial state for a new conversation.
    
    Returns:
        A fresh State instance with default values
    """
    return {
        "messages": [],
        "query_type": None,
        "confidence": None,
        "retrieved_contexts": [],
        "onboarding_progress": None,
        "knowledge_gaps": [],
        "escalated": False,
        "escalation_reason": None,
        "human_response": None,
    }