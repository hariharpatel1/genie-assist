"""
Code explorer tool for the onboarding agent.
"""
import logging
from typing import Dict, List, Optional, Type

from langchain.tools import BaseTool
from langchain_core.documents import Document
from langchain_core.tools import Tool

from retrivers.code_retriever import CodeRetriever

logger = logging.getLogger(__name__)

class CodeExplorerTools:
    """
    Provides tools for exploring and searching code repositories.
    """
    
    def __init__(self, code_retriever: Optional[CodeRetriever] = None):
        """
        Initialize the code explorer tools.
        
        Args:
            code_retriever: Optional code retriever instance to use
        """
        self.code_retriever = code_retriever or CodeRetriever()
        logger.info("Initialized code explorer tools")
    
    def search_code(self, query: str, k: int = 5) -> Dict[str, List[Dict]]:
        """
        Search for code relevant to the query.
        
        Args:
            query: The search query
            k: Number of results to return
        
        Returns:
            Dictionary with search results
        """
        docs = self.code_retriever.search(query, k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "language": doc.metadata.get("language", "unknown"),
                "repo": doc.metadata.get("repo", "unknown"),
                "file_path": doc.metadata.get("source", "unknown"),
                "relevance_score": doc.metadata.get("relevance_score", 0),
            })
        
        return {"results": results}
    
    def search_code_by_language(self, query: str, language: str, k: int = 5) -> Dict[str, List[Dict]]:
        """
        Search for code in a specific language.
        
        Args:
            query: The search query
            language: Programming language to filter by
            k: Number of results to return
        
        Returns:
            Dictionary with search results
        """
        docs = self.code_retriever.search_by_language(query, language, k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "language": doc.metadata.get("language", language),
                "repo": doc.metadata.get("repo", "unknown"),
                "file_path": doc.metadata.get("source", "unknown"),
                "relevance_score": doc.metadata.get("relevance_score", 0),
            })
        
        return {"results": results}
    
    def get_tools(self) -> List[Tool]:
        """
        Get the list of code explorer tools.
        
        Returns:
            List of LangChain tools
        """
        return [
            Tool.from_function(
                func=self.search_code,
                name="search_code",
                description=(
                    "Search for code snippets that match a query. Useful for finding "
                    "specific functionality or understanding how certain features are implemented."
                ),
                return_direct=False,
            ),
            Tool.from_function(
                func=self.search_code_by_language,
                name="search_code_by_language",
                description=(
                    "Search for code snippets in a specific programming language. "
                    "Provide the query and the language name (e.g., 'python', 'javascript')."
                ),
                return_direct=False,
            ),
        ]