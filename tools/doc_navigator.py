"""
Document navigator tool for exploring and retrieving documentation.
"""
import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.tools import Tool

from retrieval.document_retriever import DocumentRetriever

logger = logging.getLogger(__name__)

class DocNavigatorTools:
    """
    Provides tools for navigating and searching documentation.
    """
    
    def __init__(self, document_retriever: Optional[DocumentRetriever] = None):
        """
        Initialize the document navigator tools.
        
        Args:
            document_retriever: Optional document retriever instance to use
        """
        self.document_retriever = document_retriever or DocumentRetriever()
        logger.info("Initialized document navigator tools")
    
    def search_documentation(self, query: str, k: int = 5) -> Dict[str, List[Dict]]:
        """
        Search for documentation relevant to the query.
        
        Args:
            query: The search query
            k: Number of results to return
        
        Returns:
            Dictionary with search results
        """
        docs = self.document_retriever.search(query, k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source_type": doc.metadata.get("source_type", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": doc.metadata.get("relevance_score", 0),
            })
        
        return {"results": results}
    
    def search_specific_docs(self, query: str, doc_type: str, k: int = 5) -> Dict[str, List[Dict]]:
        """
        Search for documentation of a specific type.
        
        Args:
            query: The search query
            doc_type: Type of documentation to filter by (e.g., "document", "pdf")
            k: Number of results to return
        
        Returns:
            Dictionary with search results
        """
        filter_criteria = {"source_type": doc_type}
        docs = self.document_retriever.search(query, k, filter_criteria)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source_type": doc.metadata.get("source_type", doc_type),
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": doc.metadata.get("relevance_score", 0),
            })
        
        return {"results": results}
    
    def get_tools(self) -> List[Tool]:
        """
        Get the list of documentation navigator tools.
        
        Returns:
            List of LangChain tools
        """
        return [
            Tool.from_function(
                func=self.search_documentation,
                name="search_documentation",
                description=(
                    "Search for documentation that matches a query. Useful for finding "
                    "information about company processes, guidelines, and best practices."
                ),
                return_direct=False,
            ),
            Tool.from_function(
                func=self.search_specific_docs,
                name="search_specific_docs",
                description=(
                    "Search for documentation of a specific type. "
                    "Provide the query and the document type (e.g., 'document', 'pdf')."
                ),
                return_direct=False,
            ),
        ]