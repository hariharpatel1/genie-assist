"""
Vector database interactions for the knowledge base.
"""
import logging
import os
from typing import Dict, List, Optional, Union

import chromadb
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from cfg.settings import settings

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Knowledge base for the onboarding agent using ChromaDB as the vector store.
    """
    
    def __init__(self, collection_name: str = "onboarding_knowledge"):
        """
        Initialize the knowledge base.
        
        Args:
            collection_name: Name of the ChromaDB collection to use
        """
        self.collection_name = collection_name
        self.persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        
        # Create embeddings model
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_EMBEDDINGS_DEPLOYMENT_NAME,
            api_key=settings.AZURE_EMBEDDINGS_API_KEY,
            azure_endpoint=settings.AZURE_EMBEDDINGS_ENDPOINT,
            api_version=settings.AZURE_EMBEDDINGS_API_VERSION,
            chunk_size=512,  # Explicitly setting a value
        )
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize vector store
        self.vectorstore = self._initialize_vectorstore()

        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        
        logger.info(f"Initialized knowledge base with collection '{collection_name}'")
    
    def _initialize_vectorstore(self) -> VectorStore:
        """
        Initialize the vector store.
        
        Returns:
            The initialized vector store.
        """
        try:
            # Try to load existing DB
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {e}. Creating new one.")
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
    
    def add_documents(self, documents: List[Document], source_type: str) -> None:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of documents to add
            source_type: Type of source (e.g., "code", "document", "pdf")
        """
        try:
            # Add source_type to metadata
            for doc in documents:
                if "source_type" not in doc.metadata:
                    doc.metadata["source_type"] = source_type
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vectorstore.add_documents(chunks)
            #self.vectorstore.persist()
            
            logger.info(f"Added {len(chunks)} document chunks to the knowledge base")
        except Exception as e:
            logger.error(f"Error adding documents to knowledge base: {e}")
            raise
    
    def search(
        self, 
        query: str, 
        k: int = 5, 
        filter_criteria: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search the knowledge base for relevant documents.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_criteria: Optional filter criteria for the search
        
        Returns:
            List of relevant documents
        """
        try:
            # Search with optional filters
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_criteria
            )
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def search_by_type(
        self, 
        query: str, 
        source_type: str, 
        k: int = 5
    ) -> List[Document]:
        """
        Search the knowledge base for documents of a specific type.
        
        Args:
            query: The search query
            source_type: Type of source to filter by
            k: Number of results to return
        
        Returns:
            List of relevant documents
        """
        filter_criteria = {"source_type": source_type}
        return self.search(query, k, filter_criteria)
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary of statistics
        """
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {"error": str(e)}

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> bool:
        """
        Delete documents from the knowledge base by their IDs.
        
        Args:
            ids: List of document IDs to delete. If None, deletes all documents.
            **kwargs: Additional keyword arguments that might be used by the implementation.
            
        Returns:
            bool: True if deletion is successful, False otherwise.
        """
        try:
            if ids is None:
                # Delete all documents in the collection
                logger.info(f"Deleting all documents from collection '{self.collection_name}'")
                # Use get_collection to avoid the deprecation warning
                client = chromadb.PersistentClient(path=self.persist_directory)
                try:
                    collection = client.get_collection(name="onboarding_docs")
                    # Get all document IDs
                    collection_ids = collection.get()["ids"]
                    logger.info(f"Found {(collection_ids)} documents in the collection")
                    if collection_ids:
                        collection.delete(collection_ids)
                    logger.info(f"Deleted all {len(collection_ids)} documents from the collection")
                except Exception as e:
                    logger.warning(f"Error accessing collection directly: {e}")
                    # Fallback to vectorstore implementation if available
                    if hasattr(self.vectorstore, "_collection"):
                        self.vectorstore._collection.delete()
                        logger.info("Deleted all documents using vectorstore collection reference")
            else:
                # Delete specific documents by ID
                logger.info(f"Deleting {len(ids)} documents from collection '{self.collection_name}'")
                # Use the vectorstore's delete method directly
                self.vectorstore.delete(ids)
                
            # Persist changes
            self.vectorstore.persist()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents from knowledge base: {e}")
            return False

# define global variables
knowledge_base = KnowledgeBase()