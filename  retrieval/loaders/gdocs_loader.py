"""
Google Docs loader for document retrieval.
"""
import logging
import os
from typing import List, Optional

from langchain_community.document_loaders import GoogleDriveLoader
from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)

class GoogleDocsLoader:
    """
    Loads documents from Google Docs for the knowledge base.
    """
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize the Google Docs loader.
        
        Args:
            credentials_path: Path to the Google credentials JSON file
        """
        self.credentials_path = credentials_path or settings.GOOGLE_APPLICATION_CREDENTIALS
        
        # Validate credentials path
        if not self.credentials_path or not os.path.exists(self.credentials_path):
            logger.warning(
                f"Google credentials file not found at {self.credentials_path}. "
                "Google Docs loading may not work properly."
            )
        
        logger.info("Initialized Google Docs loader")
    
    def load_document(self, doc_id: str) -> List[Document]:
        """
        Load a single Google Doc by ID.
        
        Args:
            doc_id: The Google Docs document ID
        
        Returns:
            List of documents with content
        """
        try:
            loader = GoogleDriveLoader(
                credentials_path=self.credentials_path,
                document_ids=[doc_id],
                use_api=True
            )
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata["source_type"] = "document"
                doc.metadata["doc_id"] = doc_id
            
            logger.info(f"Loaded document with ID {doc_id}, {len(documents)} pages")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading Google Doc {doc_id}: {e}")
            return []
    
    def load_documents(self, doc_ids: List[str]) -> List[Document]:
        """
        Load multiple Google Docs by ID.
        
        Args:
            doc_ids: List of Google Docs document IDs
        
        Returns:
            List of documents with content
        """
        all_documents = []
        for doc_id in doc_ids:
            docs = self.load_document(doc_id)
            all_documents.extend(docs)
        
        return all_documents
    
    def load_from_folder(self, folder_id: str) -> List[Document]:
        """
        Load all documents from a Google Drive folder.
        
        Args:
            folder_id: The Google Drive folder ID
        
        Returns:
            List of documents with content
        """
        try:
            loader = GoogleDriveLoader(
                credentials_path=self.credentials_path,
                folder_id=folder_id,
                recursive=True,
                use_api=True
            )
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata["source_type"] = "document"
                doc.metadata["folder_id"] = folder_id
            
            logger.info(f"Loaded {len(documents)} documents from folder {folder_id}")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading documents from folder {folder_id}: {e}")
            return []