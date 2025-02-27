"""
Main onboarding agent implementation using LangGraph.

This module implements a state-based agent for onboarding new team members and
answering questions about code, documentation, and company processes.
The agent follows LangGraph best practices with explicit state types, 
typed nodes, and a clear graph structure.
"""
import logging
import sys
from datetime import datetime
from typing import Annotated, Dict, List, Optional, TypedDict, Union, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt, maybe_callable

from agents.state import KnowledgeGap, OnboardingProgress, RetrievedContext, State, create_initial_state
from config.settings import settings
from retrieval.code_retriever import CodeRetriever
from retrieval.document_retriever import DocumentRetriever
from tools.code_explorer import CodeExplorerTools
from tools.doc_navigator import DocNavigatorTools
from tools.human_escalation import HumanEscalationTools
from tools.onboarding_guide import OnboardingGuideTools
from utils.llm_utils import get_azure_openai_chat_model

logger = logging.getLogger(__name__)

class OnboardingAgent:
    """
    LangGraph-based agent for team onboarding and documentation Q&A.
    
    This agent uses a state machine approach to handle different types of user queries:
    - Documentation questions
    - Code exploration
    - Onboarding guidance
    - Human expert escalation
    """
    
    def __init__(self):
        """Initialize the onboarding agent."""
        # Initialize LLM
        self.llm = get_azure_openai_chat_model()
        
        # Initialize tools
        self.code_explorer_tools = CodeExplorerTools()
        self.doc_navigator_tools = DocNavigatorTools()
        self.human_escalation_tools = HumanEscalationTools()
        self.onboarding_guide_tools = OnboardingGuideTools()
        
        # Combine all tools
        self.tools = (
            self.code_explorer_tools.get_tools() +
            self.doc_navigator_tools.get_tools() +
            self.human_escalation_tools.get_tools() +
            self.onboarding_guide_tools.get_tools()
        )
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create the graph
        self.checkpoint_saver = MemorySaver()
        self.graph = self._build_graph()
        
        logger.info("Initialized onboarding agent with LangGraph")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        # Create graph builder
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("query_analyzer", self._query_analyzer)
        graph_builder.add_node("document_retriever", self._document_retriever)
        graph_builder.add_node("code_explorer", self._code_explorer)
        graph_builder.add_node("answer_generator", self._answer_generator)
        graph_builder.add_node("human_escalation", self._human_escalation)
        graph_builder.add_node("onboarding_guide", self._onboarding_guide)
        
        # Create the tool execution node using LangGraph's prebuilt ToolNode
        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)
        
        # Add edges
        # From start to query_analyzer
        graph_builder.add_edge(START, "query_analyzer")
        
        # Query analyzer routes to different nodes based on query type
        graph_builder.add_conditional_edges(
            "query_analyzer",
            self._route_query,
            {
                "documentation": "document_retriever",
                "code": "code_explorer",
                "onboarding": "onboarding_guide",
                "escalation": "human_escalation",
                "tools": "tools",
            }
        )
        
        # Document retriever and code explorer route to answer generator
        graph_builder.add_edge("document_retriever", "answer_generator")
        graph_builder.add_edge("code_explorer", "answer_generator")
        graph_builder.add_edge("tools", "answer_generator")
        
        # Answer generator routes to human escalation if confidence is low
        graph_builder.add_conditional_edges(
            "answer_generator",
            self._check_confidence,
            {
                "complete": END,
                "escalate": "human_escalation",
            }
        )
        
        # Human escalation and onboarding guide end the graph
        graph_builder.add_edge("human_escalation", END)
        graph_builder.add_edge("onboarding_guide", END)
        
        # Compile and return the graph
        return graph_builder.compile(checkpointer=self.checkpoint_saver)
    
    def _query_analyzer(self, state: State) -> Dict:
        """
        Analyze the user query to determine its type and how to route it.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with query type and confidence
        """
        # Get the most recent user message
        if not state["messages"]:
            return state
        
        messages = state["messages"]
        latest_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not latest_msg:
            return state
        
        # Define the prompt for query analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
             You are an expert query analyzer for a technical onboarding assistant.
             Analyze the user's query and determine its category. Categories are:
             
             - documentation: Questions about processes, guidelines, etc. that would be found in documentation
             - code: Questions about code, implementation details, architecture, etc.
             - onboarding: Requests related to the onboarding process or getting started
             - escalation: Complex questions that likely need human expertise
             - tools: Specific requests to use a tool function directly
             
             Respond with a JSON object containing:
             - query_type: The category of the query (one of the above)
             - confidence: Your confidence in this classification (0.0 to 1.0)
             - explanation: Brief explanation of your classification
             """),
            ("user", "{query}")
        ])
        
        # Create a chain for query analysis
        chain = (
            {"query": RunnablePassthrough()} 
            | prompt 
            | self.llm 
            | JsonOutputParser()
        )
        
        # Run the chain
        query = latest_msg.content
        try:
            result = chain.invoke(query)
            
            # Add a message to explain routing (for transparency)
            messages.append(
                AIMessage(
                    content=f"I'll help you with your {result.get('query_type', 'question')}."
                )
            )
            
            # Return state with query type and confidence
            return {
                **state,
                "query_type": result.get("query_type", "documentation"),
                "confidence": result.get("confidence", 0.7),
                "messages": messages,
            }
        
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {
                **state,
                "query_type": "documentation",  # Default to documentation
                "confidence": 0.5,
            }
    
    def _route_query(self, state: State) -> str:
        """
        Route the query to the appropriate node based on its type.
        
        Args:
            state: Current state
        
        Returns:
            Next node to route to
        """
        query_type = state.get("query_type", "documentation")
        confidence = state.get("confidence", 0.0)
        
        # If confidence is very low, escalate to human
        if confidence < 0.4:
            return "escalation"
        
        # Otherwise route based on query type
        if query_type in ["documentation", "code", "onboarding", "escalation", "tools"]:
            return query_type
        
        # Default to documentation for unrecognized types
        return "documentation"
    
    def _document_retriever(self, state: State) -> Dict:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with retrieved contexts
        """
        if not state["messages"]:
            return state
        
        # Get the most recent user message
        messages = state["messages"]
        latest_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not latest_msg:
            return state
        
        query = latest_msg.content
        
        # Use the document navigator to search for relevant content
        doc_retriever = DocumentRetriever()
        docs = doc_retriever.search(query, k=5)
        
        # Format the results
        retrieved_contexts = []
        
        for doc in docs:
            retrieved_contexts.append(
                RetrievedContext(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    source_type=doc.metadata.get("source_type", "document"),
                    relevance_score=doc.metadata.get("relevance_score", 0.0),
                    metadata=doc.metadata
                ).model_dump()
            )
        
        # Add AI message acknowledging the document search
        messages.append(
            AIMessage(
                content=f"I've searched through our documentation to find information related to your query."
            )
        )
        
        # Return updated state
        return {
            **state,
            "retrieved_contexts": retrieved_contexts,
            "messages": messages,
        }
    
    def _code_explorer(self, state: State) -> Dict:
        """
        Explore code repositories based on the query.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with retrieved contexts
        """
        if not state["messages"]:
            return state
        
        # Get the most recent user message
        messages = state["messages"]
        latest_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not latest_msg:
            return state
        
        query = latest_msg.content
        
        # Use the code explorer to search for relevant code
        code_retriever = CodeRetriever()
        docs = code_retriever.search(query, k=5)
        
        # Format the results
        retrieved_contexts = []
        
        for doc in docs:
            retrieved_contexts.append(
                RetrievedContext(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    source_type="code",
                    relevance_score=doc.metadata.get("relevance_score", 0.0),
                    metadata=doc.metadata
                ).model_dump()
            )
        
        # Add AI message acknowledging the code search
        messages.append(
            AIMessage(
                content=f"I've searched through our codebase to find relevant code snippets for your query."
            )
        )
        
        # Return updated state
        return {
            **state,
            "retrieved_contexts": retrieved_contexts,
            "messages": messages,
        }
    
    def _answer_generator(self, state: State) -> Dict:
        """
        Generate an answer based on retrieved contexts.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with answer message
        """
        if not state["messages"]:
            return state
        
        # Get the most recent user message
        messages = state["messages"]
        latest_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not latest_msg:
            return state
        
        query = latest_msg.content
        retrieved_contexts = state.get("retrieved_contexts", [])
        
        # If no contexts were retrieved, consider escalation
        if not retrieved_contexts:
            messages.append(
                AIMessage(
                    content=(
                        "I couldn't find specific information to answer your question. "
                        "I'll need to escalate this to a human expert."
                    )
                )
            )
            return {
                **state,
                "confidence": 0.0,  # Force escalation
                "messages": messages,
            }
        
        # Prepare context for the LLM
        context_text = "\n\n".join([
            f"Source: {ctx.get('source', 'Unknown')}\n"
            f"Content: {ctx.get('content', 'No content')}"
            for ctx in retrieved_contexts
        ])
        
        # Define the prompt for answer generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
             You are a helpful technical onboarding assistant. Use the following retrieved 
             information to answer the user's question. If the information doesn't fully
             answer the question, be honest about limitations and suggest next steps.
             
             Retrieved information:
             {context}
             """),
            ("user", "{query}")
        ])
        
        # Create a chain for answer generation
        chain = prompt | self.llm
        
        # Run the chain
        try:
            result = chain.invoke({"context": context_text, "query": query})
            
            # Add the answer to messages
            messages.append(result)
            
            # Assess confidence - a simple heuristic based on retrieved context relevance
            confidence = 0.0
            for ctx in retrieved_contexts:
                confidence = max(confidence, ctx.get("relevance_score", 0.0))
            
            # Cap at 0.9 to leave room for uncertainty
            confidence = min(confidence, 0.9)
            
            # Return updated state
            return {
                **state,
                "messages": messages,
                "confidence": confidence,
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            messages.append(
                AIMessage(
                    content=(
                        "I'm sorry, I encountered an error while trying to generate an answer. "
                        "I'll need to escalate this to a human expert."
                    )
                )
            )
            return {
                **state,
                "confidence": 0.0,  # Force escalation
                "messages": messages,
            }
    
    def _check_confidence(self, state: State) -> str:
        """
        Check if the confidence is high enough to complete or if we should escalate.
        
        Args:
            state: Current state
        
        Returns:
            Next node to route to
        """
        confidence = state.get("confidence", 0.0)
        
        # If confidence is below threshold, escalate
        if confidence < 0.6:
            return "escalate"
        
        # Otherwise, we're done
        return "complete"
    
    def _human_escalation(self, state: State) -> Dict:
        """
        Handle escalation to a human expert.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with human response
        """
        if not state["messages"]:
            return state
        
        # Get the most recent user message
        messages = state["messages"]
        latest_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not latest_msg:
            return state
        
        query = latest_msg.content
        
        # Prepare context from retrieved information
        retrieved_contexts = state.get("retrieved_contexts", [])
        context_text = "\n\n".join([
            f"Source: {ctx.get('source', 'Unknown')}\n"
            f"Content: {ctx.get('content', 'No content')}"
            for ctx in retrieved_contexts
        ]) if retrieved_contexts else "No relevant information found."
        
        # Use the human escalation tool
        try:
            escalation_result = self.human_escalation_tools.escalate_to_human(
                query=query,
                context=context_text
            )
            
            # Add the human response to messages
            if escalation_result.get("escalated", False):
                messages.append(
                    AIMessage(
                        content=(
                            f"I've consulted with {escalation_result.get('expert', 'a human expert')} "
                            f"who provided the following response:\n\n"
                            f"{escalation_result.get('response', 'No response provided.')}"
                        )
                    )
                )
            else:
                messages.append(
                    AIMessage(
                        content=(
                            "I attempted to escalate your question to a human expert, but there was an issue. "
                            "Please try again later or rephrase your question."
                        )
                    )
                )
            
            # Return updated state
            return {
                **state,
                "messages": messages,
                "escalated": escalation_result.get("escalated", False),
                "escalation_reason": "Low confidence in automated response",
                "human_response": escalation_result.get("response", None),
            }
        
        except Exception as e:
            logger.error(f"Error during human escalation: {e}")
            messages.append(
                AIMessage(
                    content=(
                        "I attempted to escalate your question to a human expert, but encountered an error. "
                        "Please try again later or rephrase your question."
                    )
                )
            )
            return {
                **state,
                "messages": messages,
                "escalated": False,
                "escalation_reason": f"Error: {str(e)}",
            }
    
    def _onboarding_guide(self, state: State) -> Dict:
        """
        Handle onboarding guide interactions.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with onboarding information
        """
        # Import here to avoid circular imports
        from datetime import datetime
        
        if not state["messages"]:
            return state
        
        # Get the most recent user message
        messages = state["messages"]
        latest_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not latest_msg:
            return state
        
        query = latest_msg.content
        
        # Check if we already have onboarding progress
        onboarding_progress = state.get("onboarding_progress", None)
        
        if not onboarding_progress:
            # Define the prompt for extracting user info
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                 Extract user information from the message for onboarding.
                 If the information is not provided, use reasonable defaults.
                 Respond with a JSON object containing:
                 - user_id: A unique identifier (use a timestamp if not provided)
                 - user_name: The user's name (use "New Team Member" if not provided)
                 - role: The user's role (use "Developer" if not provided)
                 """),
                ("user", "{query}")
            ])
            
            # Create a chain for user info extraction
            chain = prompt | self.llm | JsonOutputParser()
            
            try:
                # Extract user info
                user_info = chain.invoke({"query": query})
                
                # Create onboarding plan
                result = self.onboarding_guide_tools.create_onboarding_plan(
                    user_id=user_info.get("user_id", f"user_{int(datetime.now().timestamp())}"),
                    user_name=user_info.get("user_name", "New Team Member"),
                    role=user_info.get("role", "Developer")
                )
                
                # Update onboarding progress
                onboarding_progress = result.get("onboarding_progress", {})
                
                # Add welcome message
                messages.append(
                    AIMessage(
                        content=(
                            f"Welcome to the team, {user_info.get('user_name', 'New Team Member')}! "
                            f"I've created an onboarding plan for you as a {user_info.get('role', 'Developer')}. "
                            f"Let's start with your first step: **{onboarding_progress.get('current_step', 'Getting started')}**. "
                            f"\n\nWhat would you like to know about this step?"
                        )
                    )
                )
                
                # Return updated state
                return {
                    **state,
                    "messages": messages,
                    "onboarding_progress": onboarding_progress,
                }
            
            except Exception as e:
                logger.error(f"Error creating onboarding plan: {e}")
                messages.append(
                    AIMessage(
                        content=(
                            "I'm sorry, I encountered an error while setting up your onboarding plan. "
                            "Let's try again. Could you please provide your name and role?"
                        )
                    )
                )
                return {
                    **state,
                    "messages": messages,
                }
        
        else:
            # We have existing onboarding progress
            # Check if the user is asking about the current step or trying to complete it
            
            # Define the prompt for understanding the onboarding request
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                 Analyze the user's message in the context of their onboarding.
                 Their current onboarding step is: {current_step}
                 
                 Determine if they are:
                 1. Asking for information about the current step
                 2. Indicating they have completed the current step
                 3. Asking about something else
                 
                 Respond with a JSON object containing:
                 - request_type: "info", "complete", or "other"
                 - explanation: Brief explanation of your classification
                 """),
                ("user", "{query}")
            ])
            
            # Create a chain for request analysis
            chain = prompt | self.llm | JsonOutputParser()
            
            try:
                # Get current step
                current_step_info = self.onboarding_guide_tools.get_current_step(onboarding_progress)
                current_step = current_step_info.get("current_step", "No current step")
                
                # Analyze request
                request_info = chain.invoke({
                    "query": query,
                    "current_step": current_step
                })
                
                request_type = request_info.get("request_type", "info")
                
                if request_type == "complete":
                    # Complete the current step
                    result = self.onboarding_guide_tools.complete_step(onboarding_progress)
                    updated_progress = result.get("onboarding_progress", onboarding_progress)
                    next_step = result.get("next_step", None)
                    
                    if next_step:
                        messages.append(
                            AIMessage(
                                content=(
                                    f"Great job completing **{current_step}**! "
                                    f"\n\nYour next step is: **{next_step}**. "
                                    f"\n\n{self.onboarding_guide_tools._get_step_description(next_step)}"
                                    f"\n\nLet me know if you have any questions or when you've completed this step."
                                )
                            )
                        )
                    else:
                        messages.append(
                            AIMessage(
                                content=(
                                    f"Congratulations on completing **{current_step}**! "
                                    f"You've now finished all the onboarding steps. "
                                    f"\n\nIs there anything specific you'd like to learn more about now?"
                                )
                            )
                        )
                    
                    return {
                        **state,
                        "messages": messages,
                        "onboarding_progress": updated_progress,
                    }
                
                elif request_type == "info":
                    # Provide information about the current step
                    step_description = self.onboarding_guide_tools._get_step_description(current_step)
                    messages.append(
                        AIMessage(
                            content=(
                                f"Let me tell you more about **{current_step}**:\n\n"
                                f"{step_description}\n\n"
                                f"Would you like any specific details about this step? Or let me know when you're ready to mark it as complete."
                            )
                        )
                    )
                    return {
                        **state,
                        "messages": messages,
                    }
                
                else:
                    # Handle other requests using the regular tools
                    # First try to find documentation related to their query
                    doc_retriever = DocumentRetriever()
                    docs = doc_retriever.search(query, k=3)
                    
                    # Format the results
                    retrieved_contexts = []
                    
                    for doc in docs:
                        retrieved_contexts.append(
                            RetrievedContext(
                                content=doc.page_content,
                                source=doc.metadata.get("source", "Unknown"),
                                source_type=doc.metadata.get("source_type", "document"),
                                relevance_score=doc.metadata.get("relevance_score", 0.0),
                                metadata=doc.metadata
                            ).model_dump()
                        )
                    
                    # Generate a response using retrieved contexts and onboarding context
                    context_text = "\n\n".join([
                        f"Source: {ctx.get('source', 'Unknown')}\n"
                        f"Content: {ctx.get('content', 'No content')}"
                        for ctx in retrieved_contexts
                    ]) if retrieved_contexts else "No relevant information found."
                    
                    # Add onboarding context
                    context_text += f"\n\nUser is currently in the onboarding process. Current step: {current_step}"
                    
                    # Define the prompt for generating a response
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """
                         You are a helpful technical onboarding assistant. The user is currently going through
                         onboarding, with their current step being: {current_step}
                         
                         Use the following retrieved information to answer their question:
                         {context}
                         
                         Make your response helpful and supportive, relating back to their onboarding
                         process when relevant.
                         """),
                        ("user", "{query}")
                    ])
                    
                    # Create a chain for response generation
                    chain = prompt | self.llm
                    
                    # Generate response
                    response = chain.invoke({
                        "current_step": current_step,
                        "context": context_text,
                        "query": query
                    })
                    
                    messages.append(response)
                    return {
                        **state,
                        "messages": messages,
                        "retrieved_contexts": retrieved_contexts,
                    }
            
            except Exception as e:
                logger.error(f"Error handling onboarding request: {e}")
                messages.append(
                    AIMessage(
                        content=(
                            "I apologize, but I encountered an issue while processing your request. "
                            "Could you please try again or rephrase your question?"
                        )
                    )
                )
                return {
                    **state,
                    "messages": messages,
                }
    
    def invoke(self, message: str, thread_id: str = "default") -> List[Dict]:
        """
        Invoke the agent with a message.
        
        Args:
            message: User message
            thread_id: Thread ID for conversation persistence
        
        Returns:
            List of response messages
        """
        # Create input state
        input_state = create_initial_state()
        input_state["messages"] = [HumanMessage(content=message)]
        
        # Configure thread ID
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run the graph
        try:
            final_state = self.graph.invoke(input_state, config)
            
            # Extract AI messages from the final state
            ai_messages = [
                msg for msg in final_state["messages"] 
                if isinstance(msg, AIMessage)
            ]
            
            # Convert to dictionaries
            return [
                {
                    "role": "assistant",
                    "content": msg.content,
                }
                for msg in ai_messages
            ]
        
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            return [
                {
                    "role": "assistant",
                    "content": (
                        "I apologize, but I encountered an error while processing your request. "
                        "Please try again or contact support if the issue persists."
                    ),
                }
            ]
    
    def get_thread_history(self, thread_id: str = "default") -> List[Dict]:
        """
        Get the message history for a thread.
        
        Args:
            thread_id: Thread ID
        
        Returns:
            List of messages in the thread
        """
        try:
            # Configure thread ID
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get current state
            current_state = self.graph.get_state(config)
            
            if not current_state or "messages" not in current_state.values:
                return []
            
            # Convert messages to dictionaries
            return [
                {
                    "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content,
                }
                for msg in current_state.values["messages"]
            ]
        
        except Exception as e:
            logger.error(f"Error getting thread history: {e}")
            return []