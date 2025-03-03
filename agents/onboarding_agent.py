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
from langgraph.prebuilt import ToolNode

from agents.state import KnowledgeGap, OnboardingProgress, RetrievedContext, State, create_initial_state
from retrivers.code_retriever import CodeRetriever
from retrivers.document_retriever import DocumentRetriever
from tools.code_explorer import CodeExplorerTools
from tools.doc_navigator import DocNavigatorTools
from tools.human_escalation import HumanEscalationTools
from tools.onboarding_guide import OnboardingGuideTools
from utils.llm_utils import azure_llm

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
        try:
            self.llm = azure_llm
            logger.info("Successfully initialized Azure OpenAI for the onboarding agent")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI: {e}")
            raise
        
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
        
        # # Bind tools to LLM - ensure proper binding with error handling
        # try:
        #     self.llm_with_tools = self.llm.bind_tools(self.tools)
        #     logger.info("Successfully bound tools to Azure OpenAI")
        # except Exception as e:
        #     logger.error(f"Error binding tools to Azure OpenAI: {e}")
        #     raise
        
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
        graph_builder.add_node("info_extractor", self._info_extractor)
        # graph_builder.add_node("document_retriever", self._document_retriever)
        # graph_builder.add_node("code_explorer", self._code_explorer)
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
                "info_extractor": "info_extractor",
                "onboarding": "onboarding_guide",
                "escalation": "human_escalation",
                "tools": "tools",
            }
        )
        
        graph_builder.add_edge("info_extractor", "answer_generator")
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
    
    def _query_analyzer(self, state: State) -> State:
        """
        Analyze the user query to determine its type and how to route it.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with query type and confidence
        """
        logger.info("[AgentQueryAnalyzer] Analyzing user query...")
        # Get the most recent user message
        if not state.messages:
            logger.info("[AgentQueryAnalyzer] No query messages found in state")
            return state
        
        messages = state.messages
        latest_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not latest_msg:
            logger.info("[AgentQueryAnalyzer] No user message found")
            return state
        
        original_query = latest_msg.content
        logger.info(f"[AgentQueryAnalyzer] Original user query: {original_query}")

        # Step 1: Refine and enhance the user query
        query_enhancer_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a technical query enhancement for software development team.
            Your goal is to take the user's original query and create an enhanced version that:
            1. Fixes any grammatical or syntax errors
            2. Adds relevant technical terminology if missing
            3. Makes the query more specific and technically accurate
            4. Structures the query for better information retrieval
            
            You should maintain the original intent of the query while making it more precise.
            
            Respond with a JSON object containing:
            {{
                "enhanced_query": "The enhanced version of the query",
                "technical_terms_added": ["list", "of", "technical", "terms", "added"],
                "search_queries": ["list", "of", "2-3", "variations", "for", "search"]
            }}
            """),
            ("user", "{query}")
        ])

        # Create a chain for query enhancement
        query_enhancer_chain = (
            {"query": RunnablePassthrough()} 
            | query_enhancer_prompt 
            | self.llm 
            | JsonOutputParser()
        )
        
        try:
            enhanced_query_result = query_enhancer_chain.invoke({"query": original_query})
            logger.info(f"[AgentQueryAnalyzer] Enhanced query result: {enhanced_query_result}")
            enhanced_query = enhanced_query_result.get("enhanced_query", original_query)
            search_queries = enhanced_query_result.get("search_queries", [enhanced_query])
            state.enhanced_query = enhanced_query
            state.search_queries = search_queries

            
            logger.info(f"[AgentQueryAnalyzer] Enhanced query: {enhanced_query}")
            logger.info(f"[AgentQueryAnalyzer] Generated search queries: {search_queries}")
            
            # Step 2: Determine the query type using the enhanced query
            query_type_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a technical query classifier for a UPI and NPCI payment systems development software engineers team.
                Analyze the user's query and classify it into the most appropriate category.
                
                Categories:
                - "documentation": Questions about UPI architecture, flows, processes, or guidelines
                - "code": Questions about implementation details, API integration, or code structure
                - "both": Questions that require both code and documentation knowledge
                - "onboarding": Questions related to new team member onboarding or training
                - "escalation": Complex questions that need human expertise
                
                Respond with a JSON object:
                {{
                    "primary_type": "One of the categories above",
                    "confidence": "A number between 0.0 and 1.0 indicating your confidence",
                    "requires_code_context": true/false,
                    "requires_doc_context": true/false,
                    "technical_focus": "The technical aspect of focus (e.g., 'API integration', 'Transaction flow')"
                }}
                """),
                ("user", "{query}")
            ])

            # Create a chain for query type analysis
            query_type_chain = (
                {"query": RunnablePassthrough()} 
                | query_type_prompt 
                | self.llm 
                | JsonOutputParser()
            )
            
            logger.info("[AgentQueryAnalyzer] Classifying query type...")
            query_type_result = query_type_chain.invoke({"query": enhanced_query})
            
            primary_type = query_type_result.get("primary_type", "both")
            confidence = query_type_result.get("confidence", 0.7)
            requires_code = query_type_result.get("requires_code_context", False)
            requires_doc = query_type_result.get("requires_doc_context", True)
            technical_focus = query_type_result.get("technical_focus", "UPI")
            
            logger.info(f"[AgentQueryAnalyzer] Query type: {primary_type}, confidence: {confidence}")
            logger.info(f"[AgentQueryAnalyzer] Requires code context: {requires_code}")
            logger.info(f"[AgentQueryAnalyzer] Requires doc context: {requires_doc}")
            logger.info(f"[AgentQueryAnalyzer] Technical focus: {technical_focus}")
            state.primary_type = primary_type
            state.requires_code_context = requires_code
            state.requires_doc_context = requires_doc
            state.technical_focus = technical_focus
            state.confidence = confidence
            
            return state.model_validate(state.model_dump())
        
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            state.primary_type = "both"
            state.confidence = 0.5
            
            return state.model_validate(state.model_dump())
    
    def _route_query(self, state: State) -> str:
        """
        Route the query to the appropriate node based on its type.
        
        Args:
            state: Current state
        
        Returns:
            Next node to route to
        """
        logger.info("[AgentQueryRouter] Routing user query...")
        query_type = state.primary_type
        confidence = state.confidence
        
        logger.info(f"[AgentQueryRouter] Query type: {query_type}, confidence: {confidence}")

        # If confidence is very low, escalate to human
        if confidence < 0.2:
            logger.info("[AgentQueryRouter] Low confidence, escalating to human")
            return "escalation"
        
        # Otherwise route based on query type
        if query_type in ["both", "documentation", "code"]:
            logger.info(f"[AgentQueryRouter] Routing to: both documents and code")
            return "info_extractor"
        
        if query_type == "onboarding":
            logger.info("[AgentQueryRouter] Routing to onboarding guide")
            return "onboarding"
        
        if query_type == "escalation":
            logger.info("[AgentQueryRouter] Routing to human escalation")
            return "escalation"
    
        # Default to human escalation for unrecognized types
        logger.info("[AgentQueryRouter] Defaulting to human escalation")
        return "escalation"
    
    def _info_extractor(self, state: State) -> State: 
        """"
        Extract information from the query based on the query type.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with retrieved contexts
        """
        logger.info("[AgentInfoExtractor] Extracting information...")
        query_type = state.primary_type
        logger.info(f"[AgentInfoExtractor] Extracting information... for query type: {query_type}")
        
        # if query_type == "documentation":
        #     return self._document_retriever(state)
        
        # if query_type == "code":
        #     return self._code_explorer(state)
        
        # if query_type == "both":
        state = self._document_retriever(state)
        state = self._code_explorer(state)
        return state.model_validate(state.model_dump())
        
        # # Default to no information extraction for other types
        # logger.info("[AgentInfoExtractor] No information extraction needed")
        # return state

    def _document_retriever(self, state: State) -> State:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with retrieved contexts
        """
        logger.info("[AgentDocumentRetriever] Retrieving documents...")
        if not state.messages:
            logger.info("[AgentDocumentRetriever] No messages found")
            return state
        
        # Get the most recent user message
        messages = state.messages
        latest_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not latest_msg:
            logger.info("[AgentDocumentRetriever] No user message found")
            return state
        
        original_query = latest_msg.content
        enhanced_query = state.enhanced_query
        search_queries = state.search_queries
        
        logger.info(f"[AgentDocumentRetriever] Enhanced query: {enhanced_query}")
        logger.info(f"[AgentDocumentRetriever] Search queries: {search_queries}")

        # Use the document navigator to search for relevant content
        doc_retriever = DocumentRetriever()
        
        # Initialize combined results
        all_docs = []

        # Use multiple search queries to get better coverage
        for idx, query in enumerate(search_queries):
            logger.info(f"[AgentDocumentRetriever] Searching with query {idx+1}: {query}")
            docs = doc_retriever.search(query, k=3)  # Retrieve fewer per query to avoid overwhelming
            all_docs.extend(docs)
        
        # Deduplicate results based on content
        unique_docs = []
        seen_content = set()

        for doc in all_docs:
            # Create a simplified version of content for deduplication
            simplified = ' '.join(doc.page_content.split()[:50]).lower()
            if simplified not in seen_content:
                seen_content.add(simplified)
                unique_docs.append(doc)
        
        # Sort by relevance score if available
        sorted_docs = sorted(
            unique_docs, 
            key=lambda x: x.metadata.get("relevance_score", 0.0), 
            reverse=True
        )
        
        # Cap at 5 documents to avoid overwhelming
        top_docs = sorted_docs
            
        for doc in top_docs:
                context = RetrievedContext(
                        content=doc.page_content,
                        source=doc.metadata.get("source", "Unknown"),
                        source_type=doc.metadata.get("source_type", "document"),
                        relevance_score=doc.metadata.get("relevance_score", 0.0),
                        metadata=doc.metadata
                    )
                state.retrieved_contexts.append(context)
                state.doc_contexts.append(context)

        logger.info(f"[AgentDocumentRetriever] Found {len(state.retrieved_contexts)} documents")
        # Add AI message acknowledging the document search
        if state.doc_contexts:
            messages.append(
                AIMessage(
                    content=f"I've found relevant documentation about your question."
                )
            )
        else:
            messages.append(
                AIMessage(
                    content=f"I searched our documentation but couldn't find specific information about your question."
                )
            )
    
        logger.info("[AgentDocumentRetriever] Done")

        # Return updated state
        return state.model_validate(state.model_dump())
    
    def _code_explorer(self, state: State) -> State:
        """
        Explore code repositories based on the query.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with retrieved contexts
        """
        logger.info("[AgentCodeExplorer] Exploring code...")
        if not state.messages:
            logger.info("[AgentCodeExplorer] No messages found")
            return state
        
        # Get the most recent user message
        messages = state.messages
        latest_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not latest_msg:
            logger.info("[AgentCodeExplorer] No user message found")
            return state
        
        original_query = latest_msg.content
        enhanced_query = state.enhanced_query
        search_queries = state.search_queries
        technical_focus = state.technical_focus

        # Add technical focus if it helps with code search
        code_search_queries = search_queries.copy()
        if technical_focus and technical_focus.lower() not in enhanced_query.lower():
            code_search_queries.append(f"{technical_focus} {enhanced_query}")
        
        logger.info(f"[AgentCodeExplorer] Enhanced query: {enhanced_query}")
        logger.info(f"[AgentCodeExplorer] Code search queries: {code_search_queries}")
        
        # Use the code explorer to search for relevant code
        code_retriever = CodeRetriever()
        
        # Initialize combined results
        all_docs = []
        
        # Use multiple search queries to get better coverage
        for idx, query in enumerate(code_search_queries):
            logger.info(f"[AgentCodeExplorer] Searching with query {idx+1}: {query}")
            docs = code_retriever.search(query, k=3)  # Retrieve fewer per query to avoid overwhelming
            all_docs.extend(docs)
            
        # Deduplicate results based on content
        unique_docs = []
        seen_content = set()
        
        for doc in all_docs:
            # Create a simplified version of content for deduplication
            # For code, we look at the first few lines which often has the class/function definition
            code_lines = doc.page_content.split('\n')
            signature = '\n'.join(code_lines[:min(3, len(code_lines))]).strip()
            
            if signature and signature not in seen_content:
                seen_content.add(signature)
                unique_docs.append(doc)
        
        # Sort by relevance score if available
        sorted_docs = sorted(
            unique_docs, 
            key=lambda x: x.metadata.get("relevance_score", 0.0), 
            reverse=True
        )
        
        # Cap at 5 code snippets to avoid overwhelming
        top_docs = sorted_docs
        
        for doc in top_docs:
            context = RetrievedContext(
                content=doc.page_content,
                source=doc.metadata.get("source", "Unknown"),
                source_type="code",
                relevance_score=doc.metadata.get("relevance_score", 0.0),
                metadata=doc.metadata
            )
            state.code_contexts.append(context)
            state.retrieved_contexts.append(context)
        
        logger.info(f"[AgentCodeExplorer] Found {len(state.code_contexts)} unique code snippets")
        
        # Add AI message acknowledging the code search
        if state.code_contexts:
            messages.append(
                AIMessage(
                    content=f"I've found relevant code snippets that should help answer your question."
                )
            )
        else:
            messages.append(
                AIMessage(
                    content=f"I searched our codebase but couldn't find specific code related to your question."
                )
            )
        
        logger.info("[AgentCodeExplorer] Done")
        # Return updated state
        return state.model_validate(state.model_dump())
    
    def _answer_generator(self, state: State) -> State:
        """
        Generate an answer based on retrieved contexts.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with answer message
        """
        logger.info("[AgentAnswerGenerator] Generating answer...")
        if not state.messages:
            logger.info("[AgentAnswerGenerator] No messages found")
            return state
        
        # Get the most recent user message
        messages = state.messages
        latest_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not latest_msg:
            logger.info("[AgentAnswerGenerator] No user message found")
            return state
        
        query = latest_msg.content
        enhanced_query = state.enhanced_query
        technical_focus = state.technical_focus
        
        # Get contexts from state
        retrieved_contexts = state.retrieved_contexts
        code_contexts = state.code_contexts
        doc_contexts = state.doc_contexts
        
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
            logger.info("[AgentAnswerGenerator] No retrieved contexts found")
            state.confidence = 0.0  # Force escalation
            state.messages = messages
            return state.model_validate(state.model_dump())
        
        # Determine which context types we have
        has_code = len(code_contexts) > 0
        has_docs = len(doc_contexts) > 0
        
        # Log context statistics
        logger.info(f"[AgentAnswerGenerator] Total contexts: {len(retrieved_contexts)}")
        logger.info(f"[AgentAnswerGenerator] Code contexts: {len(code_contexts)}")
        logger.info(f"[AgentAnswerGenerator] Doc contexts: {len(doc_contexts)}")

        # Prepare context sections with clear separation
        context_parts = []

        if has_docs:
            doc_section = "=== DOCUMENTATION CONTEXT ===\n\n"
            doc_section += "\n\n".join([
                f"Source: {ctx.source}\n"
                f"Content: {ctx.content}"
                for ctx in doc_contexts
            ])
            context_parts.append(doc_section)
    
        if has_code:
            code_section = "=== CODE CONTEXT ===\n\n"
            code_section += "\n\n".join([
                f"Source: {ctx.source}\n"
                f"Content: {ctx.content}"
                for ctx in code_contexts
            ])
            context_parts.append(code_section)
    
        # Combine context parts
        context_text = "\n\n".join(context_parts)

        # todo Only log a truncated version of the context to prevent overly large logs
        # truncated_context = context_text[:500] + ("..." if len(context_text) > 500 else "")
        truncated_context = context_text
        logger.info(f"[AgentAnswerGenerator] Context for LLM (truncated): {truncated_context}")
        has_docs = False
        # Choose the appropriate system prompt based on context types
        if has_code and has_docs:
            # Both code and documentation
            system_prompt = """
            You're a senior engineer on the UPI payment systems team with deep knowledge of the codebase and architecture. Help your teammate understand the system in a natural, conversational way.

            Be practical and direct, like you're chatting with a colleague at lunch or during a code review. Adapt your response to what would be most helpful for their specific question - don't force a rigid structure.

            A few things to keep in mind:
            • If they're asking about implementation, share relevant code snippets that illustrate your points
            • When discussing interfaces or APIs, show the actual contract/signature for clarity
            • For complex flows or architectures, include ASCII diagrams directly in your response
            • Connect documentation concepts to their actual implementation in the code
            • Point out interesting technical aspects or gotchas that might save them time
            • Include concrete examples to illustrate abstract concepts
            • When appropriate, compare against common patterns and antipatterns

            For diagrams (using formats that require no additional rendering):
            • Use SVG markup directly embedded in the response for clean, scalable diagrams
            • Create Mermaid.js diagram notation for complex flows (client-side rendering capable)
            • Use Unicode box-drawing characters for improved visual clarity over ASCII
            • For simple cases, use emoji and text formatting to create visual hierarchies
            • Include direct PlantUML text notation for sequence and class diagrams
            • Always include a text description alongside any diagram for accessibility

            Enhance your answers with:
            • Relevant performance considerations and potential bottlenecks
            • Security implications of the described functionality
            • Common debugging approaches for related issues
            • Links between different parts of the system that might not be obvious

            Your focus area is {technical_focus}, but respond to what they're actually asking about.

            Base your response on this documentation and code:
            {context}
            """
        elif has_code:
            # Code only
            system_prompt = """
            You're a senior engineer on the UPI payment systems team with deep knowledge of the codebase. Help your teammate understand the code in a natural, conversational way.

            Be practical and direct, like you're chatting with a colleague during a coding session or review. Adapt your response to what would be most helpful for their specific question - don't follow a template.

            A few things to keep in mind:
            • give a long and detailed answer and not be lazy
            • If they're asking about implementation, share relevant all the code snippets that illustrate your points
            • When discussing interfaces or APIs, show the actual contract/signature for clarity
            • For complex flows or architectures, include ASCII diagrams directly in your response
            • Point out interesting technical aspects or gotchas that might save them time
            • Include concrete examples to illustrate abstract concepts
            • When appropriate, compare against common patterns and antipatterns


            Your focus area is {technical_focus}, but respond to what they're actually asking about.

            Base your response on this code:
            {context}
            """
        elif has_docs:
            # Documentation only
            system_prompt = """
            You're a senior engineer on the UPI payment systems team with deep knowledge of the architecture and specifications. Help your teammate understand the technical details in a natural, conversational way.

            Be practical and direct, like you're chatting with a colleague during a design discussion. Adapt your response to what would be most helpful for their specific question - don't follow a rigid structure.

            A few things to keep in mind:
            • If they're asking about interfaces or APIs, show what the actual contract might look like
            • For complex flows or architectures, include ASCII diagrams directly in your response
            • If implementation guidance would help, suggest code patterns or examples
            • Highlight key technical requirements or constraints they should be aware of
            • Focus on practical understanding rather than theoretical explanations
            • Include real-world scenarios to illustrate how the documented systems behave
            • Mention common integration challenges or gotchas

            For diagrams (using formats that require no additional rendering):
            • Use SVG markup directly embedded in the response for clean, scalable diagrams
            • Create Mermaid.js diagram notation for complex flows (client-side rendering capable)
            • Use Unicode box-drawing characters for improved visual clarity over ASCII
            • For simple cases, use emoji and text formatting to create visual hierarchies
            • Include direct PlantUML text notation for sequence and class diagrams
            • Always include a text description alongside any diagram for accessibility

            Enhance your answers with:
            • SLA and performance expectations for described systems
            • Failure modes and recovery mechanisms
            • Versioning and backward compatibility considerations
            • Integration patterns with other systems
            • Monitoring and observability approaches

            Your focus area is {technical_focus}, but respond to what they're actually asking about.

            Base your response on these specifications:
            {context}
            """
        else:
            # Fallback system prompt
            system_prompt = """
            You are a specialized technical assistant for a UPI switch development team that builds microservice-based applications.
            
            When answering questions:
            • Provide concise but comprehensive responses
            • Include ASCII diagrams directly in your response when explaining complex concepts
            • Use concrete examples to illustrate abstract ideas
            • Highlight potential pitfalls and best practices
            • Consider performance, security, and maintainability implications
            • Reference relevant industry standards or common patterns when applicable
            
            For diagrams (using formats that require no additional rendering):
            • Use SVG markup directly embedded in the response for clean, scalable diagrams
            • Create Mermaid.js diagram notation for complex flows (client-side rendering capable)
            • Use Unicode box-drawing characters for improved visual clarity over ASCII
            • For simple cases, use emoji and text formatting to create visual hierarchies
            • Include direct PlantUML text notation for sequence and class diagrams
            • Always include a text description alongside any diagram for accessibility
            
            Analyze the following retrieved information to help answer the team member's question:
            {context}
            """
        
        # Define the prompt for answer generation with the appropriate system prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        # Create a chain for answer generation with reliable error handling
        try:
            # Run the generation with Azure OpenAI
            result = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"[AgentAnswerGenerator] Attempt {attempt+1}/{max_retries} to generate answer")
                    chain = prompt | self.llm
                    result = chain.invoke({
                        "context": context_text, 
                        "query": enhanced_query if enhanced_query else query,
                        "technical_focus": technical_focus
                    })
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"[AgentAnswerGenerator] Attempt {attempt+1} failed: {e}. Retrying...")
                    else:
                        raise
            
            if result:
                # Add the answer to messages
                messages.append(result)
                
                # Assess confidence - a simple heuristic based on retrieved context relevance
                confidence = 0.0
                for ctx in retrieved_contexts:
                    confidence = max(confidence, state.confidence)
                
                # Boost confidence if we have both code and documentation contexts
                if has_code and has_docs:
                    confidence = min(confidence + 0.2, 0.9)  # Cap at 0.9
                
                logger.info(f"[AgentAnswerGenerator] Answer generated with confidence: {confidence}")
                state.confidence = confidence

                # Return updated state
                return state.model_validate(state.model_dump())
            else:
                raise Exception("Failed to generate answer after all retries")
        
        except Exception as e:
            logger.error(f"[AgentAnswerGenerator] Error generating answer: {e}")
            messages.append(
                AIMessage(
                    content=(
                        "I'm sorry, I encountered an error while trying to generate an answer. "
                        "Let me try to provide a simple response based on what I know."
                    )
                )
            )
            
            # Fallback - generate a basic response without using the context
            try:
                fallback_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful technical assistant. Provide a general response to the user's question."),
                    ("user", "{query}")
                ])
                
                fallback_result = (fallback_prompt | self.llm).invoke({"query": query})
                messages.append(fallback_result)
                state.confidence = 0.3  # Low confidence but not forcing escalation
                state.messages = messages
                return state.model_validate(state.model_dump())
            except Exception as fallback_error:
                logger.error(f"[AgentAnswerGenerator] Fallback response also failed: {fallback_error}")
                messages.append(
                    AIMessage(
                        content=(
                            "I'm sorry, I'm having trouble generating a response right now. "
                            "Please try again later or contact support."
                        )
                    )
                )
                state.confidence = 0.0  # Force escalation
                state.messages = messages
                return state.model_validate(state.model_dump())

    
    def _check_confidence(self, state: State) -> str:
        """
        Check if the confidence is high enough to complete or if we should escalate.
        
        Args:
            state: Current state
        
        Returns:
            Next node to route to
        """
        logger.info(f"[AgentConfidenceChecker] Checking confidence. {state.confidence}")
        # confidence = state.get("confidence", 0.0)
        
        # If confidence is below threshold, escalate
        # if confidence < 0:
        #    return "escalate"
        
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
        logger.info("[AgentHumanEscalation] Escalating to human expert...")
        if not state.messages:
            logger.info("[AgentHumanEscalation] No messages found")
            return state
        
        # Get the most recent user message
        messages = state.messages
        latest_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not latest_msg:
            logger.info("[AgentHumanEscalation] No user message found")
            return state
        
        query = latest_msg.content
        
        # Prepare context from retrieved information
        retrieved_contexts = state.retrieved_contexts
        context_text = "\n\n".join([
            f"Source: {ctx.source}\n"
            f"Content: {ctx.content}"
            for ctx in retrieved_contexts
        ]) if retrieved_contexts else "No relevant information found."
        
        logger.info(f"[AgentHumanEscalation] Context for human expert: {context_text}")
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
            logger.info("[AgentHumanEscalation] Escalation done")
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
    
    def _onboarding_guide(self, state: State) -> State:
        """
        Handle onboarding guide interactions.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with onboarding information
        """
        logger.info("[AgentOnboardingGuide] Handling onboarding request...")
        
        if not state.messages:
            return state
        
        # Get the most recent user message
        messages = state.messages
        latest_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        if not latest_msg:
            return state
        
        query = latest_msg.content
        
        # Check if we already have onboarding progress
        onboarding_progress = state.onboarding_progress
        logger.info(f"[AgentOnboardingGuide] Onboarding progress: {onboarding_progress}")
        
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
                
                IMPORTANT: Return only valid JSON without any comments, markdown formatting, or additional text.
                Example of valid response format:
                {{"user_id": "1697046400", "user_name": "New Team Member", "role": "Developer"}}
                """),
                ("user", "{query}")
            ])
            
            # Create a chain for user info extraction with the safe parser
            chain = prompt | self.llm | JsonOutputParser()
            
            try:
                # Extract user info
                user_info = chain.invoke({"query": query})
                
                # Create a structured onboarding plan based on the role
                onboarding_plan = self._create_structured_onboarding_plan(
                    user_id=user_info.get("user_id", f"user_{int(datetime.now().timestamp())}"),
                    user_name=user_info.get("user_name", "New Team Member"),
                    role=user_info.get("role", "Developer")
                )
                
                # Create OnboardingProgress object
                onboarding_progress = OnboardingProgress(
                    user_id=user_info.get("user_id", f"user_{int(datetime.now().timestamp())}"),
                    user_name=user_info.get("user_name", "New Team Member"),
                    role=user_info.get("role", "Developer"),
                    all_steps=onboarding_plan["all_steps"],
                    current_step=onboarding_plan["current_step"],
                    completed_steps=onboarding_plan["completed_steps"],
                    remaining_steps=onboarding_plan["remaining_steps"],
                    step_descriptions=onboarding_plan["step_descriptions"]
                )
                
                # Add welcome message
                messages.append(
                    AIMessage(
                        content=(
                            f"# Welcome to the team, {onboarding_progress.user_name}! 🎉\n\n"
                            f"I've created a personalized onboarding plan for you as a {onboarding_progress.role}. "
                            f"We'll work through this plan step-by-step to get you fully onboarded.\n\n"
                            f"## Your First Step: **{onboarding_progress.current_step}**\n\n"
                            f"{onboarding_progress.step_descriptions[onboarding_progress.current_step]}\n\n"
                            f"What specific information would you like to know about this step? Or would you like me to guide you through it?"
                        )
                    )
                )
                logger.info("[AgentOnboardingGuide] Onboarding plan created")
                
                # Update state with new information
                state.onboarding_progress = onboarding_progress
                
                # Return state
                return state
                
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
                return state
                
        else:
            # We have existing onboarding progress
            # Check if the user is asking about the current step or trying to complete it
            logger.info("[AgentOnboardingGuide] Handling onboarding request... We have existing onboarding progress")
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
                
                IMPORTANT: Return only valid JSON without any comments, markdown formatting, or additional text.
                Example of valid response format:
                {"request_type": "info", "explanation": "The user is asking for details about the current step"}
                """),
                ("user", "{query}")
            ])
            
            # Create a chain for request analysis with the safe parser
            chain = prompt | self.llm | JsonOutputParser()
            
            try:
                # Get current step from the progress object
                current_step = onboarding_progress.current_step
                
                # Analyze request
                request_info = chain.invoke({
                    "query": query,
                    "current_step": current_step
                })
                
                request_type = request_info.get("request_type", "info")
                
                if request_type == "complete":
                    # Mark current step as completed
                    completed_steps = onboarding_progress.completed_steps.copy()
                    completed_steps.append(current_step)
                    
                    # Update remaining steps
                    remaining_steps = [step for step in onboarding_progress.remaining_steps if step != current_step]
                    
                    # Determine the next step
                    next_step = remaining_steps[0] if remaining_steps else None
                    
                    # Update the onboarding progress object
                    updated_progress = OnboardingProgress(
                        user_id=onboarding_progress.user_id,
                        user_name=onboarding_progress.user_name,
                        role=onboarding_progress.role,
                        all_steps=onboarding_progress.all_steps,
                        current_step=next_step if next_step else current_step,
                        completed_steps=completed_steps,
                        remaining_steps=remaining_steps,
                        step_descriptions=onboarding_progress.step_descriptions
                    )
                    
                    if next_step:
                        # Get the description for the next step
                        next_step_description = updated_progress.step_descriptions.get(next_step, "No description available.")
                        
                        messages.append(
                            AIMessage(
                                content=(
                                    f"## Great job completing **{current_step}**! ✅\n\n"
                                    f"You're making excellent progress. Let's move on to your next step:\n\n"
                                    f"### **{next_step}**\n\n"
                                    f"{next_step_description}\n\n"
                                    f"What would you like to know about this step? Or let me know when you've completed it."
                                )
                            )
                        )
                    else:
                        messages.append(
                            AIMessage(
                                content=(
                                    f"## 🎉 Congratulations on completing **{current_step}**! \n\n"
                                    f"You've now finished all the onboarding steps in your plan. Well done!\n\n"
                                    f"Is there anything specific about the team, codebase, or processes you'd like to learn more about now?"
                                )
                            )
                        )
                    
                    state.onboarding_progress = updated_progress
                    return state
                    
                elif request_type == "info":
                    # Provide information about the current step
                    step_description = onboarding_progress.step_descriptions.get(current_step, "No description available.")
                    
                    messages.append(
                        AIMessage(
                            content=(
                                f"## More about **{current_step}**\n\n"
                                f"{step_description}\n\n"
                                f"Would you like more specific details about this step? Or let me know when you've completed it and are ready to move on."
                            )
                        )
                    )
                    
                    return state.model_validate(state.model_dump())
                    
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
                            )
                        )
                    
                    # Generate a response using retrieved contexts and onboarding context
                    context_text = "\n\n".join([
                        f"Source: {ctx.source}\n"
                        f"Content: {ctx.content}"
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
                        process when relevant. Format your response with Markdown headings and bullet points
                        where appropriate for better readability.
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
                    state.retrieved_contexts = retrieved_contexts
                    return state.model_validate(state.model_dump())
                    
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
                return state.model_validate(state.model_dump())
   
    def _create_structured_onboarding_plan(self, user_id: str, user_name: str, role: str) -> dict:
        """
        Create a structured onboarding plan based on the role.
        
        Args:
            user_id: User ID
            user_name: User name
            role: User role
            
        Returns:
            Dictionary with the onboarding plan details
        """
        # Define the UPI development team onboarding steps based on the paste-2.txt structure
        all_steps = [
            "Initial Setup and Access",
            "Product Overview",
            "Tech Stack Overview",
            "First Hands-On Task",
            "Switch Architecture",
            "Key APIs and Flows",
            "Environment Setup and Debugging Tools",
            "Contribution Guidelines",
            "Deep Dive into Features"
        ]
        
        # Step descriptions
        step_descriptions = {
            "Initial Setup and Access": (
                "Let's get all the necessary access and setup completed:\n\n"
                "- Github access for the repositories\n"
                "- Grafana access for monitoring\n"
                "- Coralogix access for logs\n"
                "- Harbor access for container registry\n"
                "- VPN access for secure connections\n"
                "- JarvisKube access for deployment\n"
                "- Jira board access for task management\n\n"
                "You should contact #tech_it for most of these accesses. For reference, you can also check: "
                "https://alpha.razorpay.com/repo/eng_onboarding-2"
            ),
            "Product Overview": (
                "Learn about the UPI infrastructure and product suite:\n\n"
                "- Understand business workflows and main use cases\n"
                "- Study the UPI Switch concept and architecture\n"
                "- Review the UPI 2.0 technical specification document\n"
                "- Understand how UPI works end-to-end\n"
                "- Review the acquiring side Go-Live concept note\n\n"
                "This will help you understand what we're building and why."
            ),
            "Tech Stack Overview": (
                "Familiarize yourself with our technology stack:\n\n"
                "- Go programming language (complete the Go tour: https://go.dev/tour/welcome/1)\n"
                "- Git version control (https://learngitbranching.js.org/)\n"
                "- GORM for database interactions\n"
                "- Kubernetes for container orchestration\n"
                "- gRPC for service communication\n"
                "- Domain-Driven Design principles\n"
                "- Kafka for event streaming\n\n"
                "Also review the UPI Switch V2 Architecture Walkthrough recording."
            ),
            "First Hands-On Task": (
                "Time to get your hands dirty with the code:\n\n"
                "- Clone the UPI Switch repository (https://github.com/razorpay/upi-switch)\n"
                "- Set up local development environment with Colima\n"
                "- Set up local Kafka instance\n"
                "- Run the application locally\n"
                "- Fix a small bug or implement a minor feature\n\n"
                "Follow the contribution framework guidelines in the README.md."
            ),
            "Switch Architecture": (
                "Gain a deeper understanding of the Switch architecture:\n\n"
                "- Study the package structure walkthrough recording\n"
                "- Review the events design discussion recording\n"
                "- Understand the UPI Switch design framework\n"
                "- Study the UPI domain model\n\n"
                "This will help you understand how the different components fit together."
            ),
            "Key APIs and Flows": (
                "Learn about the key APIs used in the Switch:\n\n"
                "- Payments for PG flow\n"
                "- UPI Acquiring Switch - API document\n"
                "- NPCI and bank integration points\n"
                "- Request/response formats\n\n"
                "Understanding these APIs will help you work on features involving payment flows."
            ),
            "Environment Setup and Debugging Tools": (
                "Learn about our deployment and debugging infrastructure:\n\n"
                "- Understand Kubemanifest for Kubernetes deployments\n"
                "- Learn about self-serve infrastructure provisioning\n"
                "- Get familiar with JarvisKube for deployments\n"
                "- Practice using Coralogix for logs and traces\n"
                "- Learn to use Grafana for metrics and monitoring\n\n"
                "These tools will help you deploy and troubleshoot issues in your code."
            ),
            "Contribution Guidelines": (
                "Learn about how we contribute to the codebase:\n\n"
                "- Testing strategy and practices\n"
                "- Code review process\n"
                "- Pull request guidelines\n"
                "- Documentation standards\n\n"
                "Following these guidelines ensures your contributions meet our quality standards."
            ),
            "Deep Dive into Features": (
                "Take on more complex tasks and get deeper into the codebase:\n\n"
                "- Work on a medium-complexity feature or bug fix\n"
                "- Review recent pull requests to understand changes\n"
                "- Meet with your mentor to discuss your progress\n"
                "- Identify areas for further learning\n\n"
                "This step helps solidify your understanding and contribution abilities."
            )
        }
        
        # Start with the first step
        current_step = all_steps[0]
        
        # Initially, all steps except the current one are remaining
        completed_steps = []
        remaining_steps = all_steps.copy()
        
        return {
            "all_steps": all_steps,
            "current_step": current_step,
            "completed_steps": completed_steps,
            "remaining_steps": remaining_steps,
            "step_descriptions": step_descriptions
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
        input_state.messages = [HumanMessage(content=message)]
        
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