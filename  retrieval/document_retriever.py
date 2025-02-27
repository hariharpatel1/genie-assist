"""
Document retrieval service that combines various document sources.
"""
import logging
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document

from config.settings import settings
from retrieval.knowledge_base import KnowledgeBase
from retrieval.loaders.gdocs_loader import GoogleDocsLoader
from retrieval.loaders.pdf_loader import PDFDocumentLoader

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """
    Retrieves documents from the knowledge base and various sources.
    """
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        """
        Initialize the document retriever.
        
        Args:
            knowledge_base: Optional knowledge base instance to use
        """
        self.knowledge_base = knowledge_base or KnowledgeBase(collection_name="onboarding_docs")
        self.gdocs_loader = GoogleDocsLoader()
        self.pdf_loader = PDFDocumentLoader()
        
        logger.info("Initialized document retriever")
    
    def search(
        self, 
        query: str, 
        k: int = 5, 
        filter_criteria: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search for documents relevant to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_criteria: Optional filter criteria for the search
        
        Returns:
            List of relevant documents
        """
        return self.knowledge_base.search(query, k, filter_criteria)
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Search specifically for documents (not code or other types).
        
        Args:
            query: The search query
            k: Number of results to return
        
        Returns:
            List of relevant documents
        """
        return self.knowledge_base.search_by_type(query, source_type="document", k=k)
    
    def load_and_index_google_docs(self, doc_ids: Optional[List[str]] = None) -> int:
        """
        Load and index Google Docs into the knowledge base.
        
        Args:
            doc_ids: Optional list of document IDs to load. If None, uses the IDs from settings.
        
        Returns:
            Number of documents indexed
        """
        try:
            doc_ids_to_load = doc_ids or settings.GOOGLE_DOCS_IDS
            
            if not doc_ids_to_load:
                logger.warning("No Google Docs IDs provided for indexing")
                return 0
            
            # Load documents
            documents = self.gdocs_loader.load_documents(doc_ids_to_load)
            
            if not documents:
                logger.warning("No documents were loaded from Google Docs")
                return 0
            
            # Add to knowledge base
            self.knowledge_base.add_documents(documents, source_type="document")
            
            logger.info(f"Indexed {len(documents)} Google Docs documents")
            return len(documents)
        
        except Exception as e:
            logger.error(f"Error indexing Google Docs: {e}")
            return 0
    
    def load_and_index_pdfs(self, file_paths: List[str]) -> int:
        """
        Load and index PDF files into the knowledge base.
        
        Args:
            file_paths: List of paths to PDF files
        
        Returns:
            Number of documents (pages) indexed
        """
        try:
            # Load documents
            documents = self.pdf_loader.load_documents(file_paths)
            
            if not documents:
                logger.warning("No documents were loaded from PDFs")
                return 0
            
            # Add to knowledge base
            self.knowledge_base.add_documents(documents, source_type="pdf")
            
            logger.info(f"Indexed {len(documents)} PDF pages")
            return len(documents)
        
        except Exception as e:
            logger.error(f"Error indexing PDFs: {e}")
            return 0
    
    def load_and_index_pdf_directory(self, directory_path: str, recursive: bool = True) -> int:
        """
        Load and index all PDFs in a directory.
        
        Args:
            directory_path: Path to the directory containing PDFs
            recursive: Whether to search recursively
        
        Returns:
            Number of documents (pages) indexed
        """
        try:
            # Load documents
            documents = self.pdf_loader.load_from_directory(directory_path, recursive)
            
            if not documents:
                logger.warning(f"No documents were loaded from directory {directory_path}")
                return 0
            
            # Add to knowledge base
            self.knowledge_base.add_documents(documents, source_type="pdf")
            
            logger.info(f"Indexed {len(documents)} PDF pages from directory {directory_path}")
            return len(documents)
        
        except Exception as e:
            logger.error(f"Error indexing PDFs from directory: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the document knowledge base.
        
        Returns:
            Dictionary of statistics
        """
        return self.knowledge_base.get_stats()