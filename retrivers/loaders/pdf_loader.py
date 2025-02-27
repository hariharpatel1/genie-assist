"""
PDF document loader for document retrieval.
"""
import logging
import os
from typing import Dict, List, Optional, Union
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class PDFDocumentLoader:
    """
    Loads PDF documents for the knowledge base.
    """
    
    def __init__(self, use_unstructured: bool = False):
        """
        Initialize the PDF document loader.
        
        Args:
            use_unstructured: Whether to use the UnstructuredPDFLoader instead of PyPDFLoader
                             UnstructuredPDFLoader is more accurate but slower and requires more dependencies
        """
        self.use_unstructured = use_unstructured
        logger.info(f"Initialized PDF document loader (use_unstructured={use_unstructured})")
    
    def _get_loader_for_file(self, file_path: str):
        """
        Get the appropriate loader for a file.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            PDF loader instance
        """
        if self.use_unstructured:
            return UnstructuredPDFLoader(file_path)
        else:
            return PyPDFLoader(file_path)
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single PDF document.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            List of documents with content (one per page)
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"PDF file not found: {file_path}")
                return []
            
            loader = self._get_loader_for_file(file_path)
            documents = loader.load()
            
            # Add metadata
            file_name = os.path.basename(file_path)
            for doc in documents:
                doc.metadata["source_type"] = "pdf"
                doc.metadata["file_name"] = file_name
                doc.metadata["file_path"] = file_path
            
            logger.info(f"Loaded PDF {file_path} with {len(documents)} pages")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return []
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple PDF documents.
        
        Args:
            file_paths: List of paths to PDF files
        
        Returns:
            List of documents with content
        """
        all_documents = []
        for file_path in file_paths:
            docs = self.load_document(file_path)
            all_documents.extend(docs)
        
        return all_documents
    
    def load_from_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """
        Load all PDF documents from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively in subdirectories
        
        Returns:
            List of documents with content
        """
        try:
            if not os.path.exists(directory_path):
                logger.error(f"Directory not found: {directory_path}")
                return []
            
            pdf_files = []
            directory = Path(directory_path)
            
            # Find all PDF files
            pattern = "**/*.pdf" if recursive else "*.pdf"
            for pdf_path in directory.glob(pattern):
                pdf_files.append(str(pdf_path))
            
            # Load all PDFs
            return self.load_documents(pdf_files)
        
        except Exception as e:
            logger.error(f"Error loading PDFs from directory {directory_path}: {e}")
            return []

#define global instance
pdf_loader = PDFDocumentLoader(use_unstructured=True)