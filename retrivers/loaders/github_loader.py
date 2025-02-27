"""
GitHub repository loader for code retrieval.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set

import git
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.documents import Document

from cfg.settings import settings

logger = logging.getLogger(__name__)

class GitHubRepositoryLoader:
    """
    Loads code from GitHub repositories for the knowledge base.
    """
    
    # File extensions to ignore
    IGNORE_EXTENSIONS = {
        ".pyc", ".so", ".o", ".a", ".lib", ".dll", ".dylib", ".exe", ".bin",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".ico", ".svg",
        ".zip", ".tar", ".gz", ".7z", ".rar", ".jar", ".war", ".ear",
        ".class", ".pdb", ".ilk", ".exp", ".obj", ".idb", ".mp3", ".mp4",
        ".avi", ".mov", ".wmv", ".flv", ".mkv", ".woff", ".woff2", ".ttf",
        ".eot", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"
    }
    
    # Directories to ignore
    IGNORE_DIRECTORIES = {
        ".git", ".github", "__pycache__", "node_modules", "venv", "env",
        "build", "dist", "target", "bin", "obj", ".idea", ".vscode"
    }
    
    def __init__(self, access_token: Optional[str] = None):
        """
        Initialize the GitHub repository loader.
        
        Args:
            access_token: GitHub access token for private repos
        """
        self.access_token = access_token or settings.GITHUB_ACCESS_TOKEN
        logger.info("Initialized GitHub repository loader")
    
    def _get_repo_url(self, repo_path: str) -> str:
        """
        Get the repository URL with auth token if available.
        
        Args:
            repo_path: Repository path in the format "owner/repo"
        
        Returns:
            Repository URL with auth token if available
        """
        if self.access_token:
            return f"https://{self.access_token}@github.com/{repo_path}.git"
        else:
            return f"https://github.com/{repo_path}.git"
    
    def _should_ignore_file(self, file_path: str) -> bool:
        """
        Check if a file should be ignored.
        
        Args:
            file_path: Path to the file
        
        Returns:
            True if the file should be ignored, False otherwise
        """
        # Check if file has an ignored extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() in self.IGNORE_EXTENSIONS:
            return True
        
        # Check if file is in an ignored directory
        path_parts = Path(file_path).parts
        for part in path_parts:
            if part in self.IGNORE_DIRECTORIES:
                return True
        
        return False
    
    def load_repository(
        self, 
        repo_path: str, 
        branch: str = "main",
        file_filter: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load a GitHub repository.
        
        Args:
            repo_path: Repository path in the format "owner/repo"
            branch: Branch to load
            file_filter: Optional list of file extensions to include (e.g., [".py", ".js"])
        
        Returns:
            List of documents with code content
        """
        try:
            # Create temporary directory for cloning
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Cloning repository {repo_path} to {temp_dir}")
                
                # Clone repository
                repo_url = self._get_repo_url(repo_path)
                repo = git.Repo.clone_from(repo_url, temp_dir, branch=branch, depth=1)
                
                # Set up the language parser
                suffixes = file_filter if file_filter else None
                loader = GenericLoader.from_filesystem(
                    temp_dir,
                    glob="**/*",
                    suffixes=suffixes,
                    parser=LanguageParser(parser_threshold=500)
                )
                
                # Load documents
                documents = loader.load()
                
                # Filter out ignored files
                filtered_docs = [
                    doc for doc in documents 
                    if not self._should_ignore_file(doc.metadata.get("source", ""))
                ]
                
                # Add repository metadata
                for doc in filtered_docs:
                    doc.metadata["repo"] = repo_path
                    doc.metadata["branch"] = branch
                    doc.metadata["source_type"] = "code"
                
                logger.info(f"Loaded {len(filtered_docs)} documents from repository {repo_path}")
                return filtered_docs
        
        except Exception as e:
            logger.error(f"Error loading repository {repo_path}: {e}")
            return []
    
    def load_repositories(
        self, 
        repo_paths: List[str],
        branch: str = "main",
        file_filter: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load multiple GitHub repositories.
        
        Args:
            repo_paths: List of repository paths in the format "owner/repo"
            branch: Branch to load
            file_filter: Optional list of file extensions to include
        
        Returns:
            List of documents with code content
        """
        all_documents = []
        for repo_path in repo_paths:
            docs = self.load_repository(repo_path, branch, file_filter)
            all_documents.extend(docs)
        
        return all_documents

# define global instance
github_loader = GitHubRepositoryLoader()