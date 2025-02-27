"""
Code retrieval service for searching and navigating codebases.
"""
import logging
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document

from cfg.settings import settings

from retrivers.knowledge_base import KnowledgeBase
from retrivers.loaders.github_loader import GitHubRepositoryLoader

logger = logging.getLogger(__name__)

class CodeRetriever:
    """
    Retrieves code from the knowledge base and various code repositories.
    """
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        """
        Initialize the code retriever.
        
        Args:
            knowledge_base: Optional knowledge base instance to use
        """
        self.knowledge_base = knowledge_base or KnowledgeBase(collection_name="onboarding_code")
        self.github_loader = GitHubRepositoryLoader()
        
        logger.info("Initialized code retriever")
    
    def search(
        self, 
        query: str, 
        k: int = 5, 
        filter_criteria: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search for code relevant to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_criteria: Optional filter criteria for the search
        
        Returns:
            List of relevant code documents
        """
        return self.knowledge_base.search(query, k, filter_criteria)
    
    def search_by_language(self, query: str, language: str, k: int = 5) -> List[Document]:
        """
        Search for code in a specific programming language.
        
        Args:
            query: The search query
            language: Programming language to filter by (e.g., "python", "javascript")
            k: Number of results to return
        
        Returns:
            List of relevant code documents
        """
        filter_criteria = {"language": language.lower()}
        return self.search(query, k, filter_criteria)
    
    def search_by_repo(self, query: str, repo_path: str, k: int = 5) -> List[Document]:
        """
        Search for code in a specific repository.
        
        Args:
            query: The search query
            repo_path: Repository path to filter by (e.g., "owner/repo")
            k: Number of results to return
        
        Returns:
            List of relevant code documents
        """
        filter_criteria = {"repo": repo_path}
        return self.search(query, k, filter_criteria)
    
    def load_and_index_repository(
        self, 
        repo_path: str, 
        branch: str = "main",
        file_filter: Optional[List[str]] = None
    ) -> int:
        """
        Load and index a GitHub repository into the knowledge base.
        
        Args:
            repo_path: Repository path in the format "owner/repo"
            branch: Branch to load
            file_filter: Optional list of file extensions to include
        
        Returns:
            Number of documents indexed
        """
        try:
            # Load documents
            documents = self.github_loader.load_repository(repo_path, branch, file_filter)
            
            if not documents:
                logger.warning(f"No documents were loaded from repository {repo_path}")
                return 0
            
            # Add to knowledge base
            self.knowledge_base.add_documents(documents, source_type="code")
            
            logger.info(f"Indexed {len(documents)} code files from repository {repo_path}")
            return len(documents)
        
        except Exception as e:
            logger.error(f"Error indexing repository {repo_path}: {e}")
            return 0
    
    def load_and_index_repositories(
        self, 
        repo_paths: Optional[List[str]] = None,
        branch: str = "main",
        file_filter: Optional[List[str]] = None
    ) -> int:
        """
        Load and index multiple GitHub repositories into the knowledge base.
        
        Args:
            repo_paths: Optional list of repository paths. If None, uses the repos from settings.
            branch: Branch to load
            file_filter: Optional list of file extensions to include
        
        Returns:
            Total number of documents indexed
        """
        try:
            repos_to_load = repo_paths or settings.GITHUB_REPOS
            
            if not repos_to_load:
                logger.warning("No GitHub repository paths provided for indexing")
                return 0
            
            total_indexed = 0
            for repo_path in repos_to_load:
                num_indexed = self.load_and_index_repository(repo_path, branch, file_filter)
                total_indexed += num_indexed
            
            return total_indexed
        
        except Exception as e:
            logger.error(f"Error indexing repositories: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the code knowledge base.
        
        Returns:
            Dictionary of statistics
        """
        return self.knowledge_base.get_stats()
    
# define global code retriever instance
code_retriever = CodeRetriever()