"""
State definitions for the onboarding agent LangGraph workflow.
"""

from typing import Annotated, Dict, List, Optional, Union, ClassVar

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain.tools.base import BaseTool

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
    all_steps: List[str] = Field(default_factory=list)
    
    step_descriptions: Dict[str, str] = Field(default_factory=dict)
    
    # Preferences
    preferred_docs_format: Optional[str] = None
    areas_of_interest: List[str] = Field(default_factory=list)
    
    model_config: ClassVar[dict] = {"populate_by_name": True}

class KnowledgeGap(BaseModel):
    """Represents a gap in the knowledge base detected by the system."""
    
    query: str
    timestamp: str
    category: str
    confidence_score: float
    suggested_resources: Optional[List[str]] = None
    
    model_config: ClassVar[dict] = {"populate_by_name": True}

class RetrievedContext(BaseModel):
    """Represents context retrieved from various sources."""
    
    content: str
    source: str
    source_type: str  # e.g., "code", "document", "pdf"
    relevance_score: float
    metadata: Dict = Field(default_factory=dict)
    
    model_config: ClassVar[dict] = {"populate_by_name": True}

class State(BaseModel):
    """The state maintained by the LangGraph workflow."""
    
    # Conversation tracking - using LangGraph's add_messages reducer
    messages: List[AnyMessage] = Field(default_factory=list, serialization_alias="messages")
    
    # Query processing
    query_type: Optional[str] = None
    confidence: Optional[float] = None
    enhanced_query: Optional[str] = None
    search_queries: Optional[List[str]] = None
    requires_code_context: bool = False
    requires_doc_context: bool = True
    technical_focus: Optional[str] = None
    
    # Type of retrieval information
    # code, documentation, both, escalation, onboarding
    primary_type: Optional[str] = "documentation"
    
    # Retrieved information
    retrieved_contexts: List[RetrievedContext] = Field(default_factory=list)
    code_contexts: List[RetrievedContext] = Field(default_factory=list)
    doc_contexts: List[RetrievedContext] = Field(default_factory=list)
    
    # Onboarding state
    onboarding_progress: Optional[OnboardingProgress] = None
    
    # Knowledge gaps tracking
    knowledge_gaps: List[KnowledgeGap] = Field(default_factory=list)
    
    # Human escalation tracking
    escalated: bool = False
    escalation_reason: Optional[str] = None
    human_response: Optional[str] = None
    
    
    model_config: ClassVar[dict] = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }

def create_initial_state() -> State:
    """
    Create the initial state for a new conversation.
    
    Returns:
        A fresh State instance with default values
    """
    return State(
        messages=[],
        query_type=None,
        confidence=None,
        enhanced_query=None,
        search_queries=None,
        requires_code_context=False,
        requires_doc_context=True,
        technical_focus=None,
        primary_type="documentation",
        retrieved_contexts=[],
        code_contexts=[],
        doc_contexts=[],
        onboarding_progress=None,
        knowledge_gaps=[],
        escalated=False,
        escalation_reason=None,
        human_response=None,
    )