import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import os
import pinecone
from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore as LangchainVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore as LlamaIndexPineconeStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStore as LlamaIndexVectorStore
from llama_index.core.retrievers import VectorIndexRetriever

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_ENVIRONMENT,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
    TOP_K,
    SIMILARITY_THRESHOLD
)
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for document storage and retrieval using Pinecone."""
    
    def __init__(self, namespace="inventory", embedding_service=None):
        """Initialize the vector store with Pinecone."""
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Check if index exists, if not create it
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if PINECONE_INDEX_NAME not in existing_indexes:
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric=PINECONE_METRIC,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-west-2"
                    )
                )
            
            # Get the index
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            logger.info(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
        
        self.namespace = namespace
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Initialize LangChain vector store
        self.langchain_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embedding_service.get_langchain_embeddings(),
            namespace=self.namespace,
            text_key="text"
        )
        
        # Initialize LlamaIndex vector store
        self.llama_store = LlamaIndexPineconeStore(
            pinecone_index=self.index,
            namespace=self.namespace,
            embed_model=self.embedding_service.get_llama_embeddings()
        )
        
        logger.info(f"Initialized vector store with namespace: {self.namespace}")
    
    def add_documents(self, documents, namespace=None):
        """Add documents to the vector store.
        
        Args:
            documents (list): List of (id, vector, metadata) tuples
            namespace (str, optional): Namespace to add documents to
        """
        try:
            # Upsert documents in batches
            self.index.upsert(
                vectors=documents,
                namespace=namespace
            )
            logger.info(f"Successfully added {len(documents)} documents to namespace: {namespace}")
            return [doc[0] for doc in documents]  # Return list of document IDs
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def query(self, query_vector, top_k=5, namespace=None):
        """Query the vector store.
        
        Args:
            query_vector (list): Query vector
            top_k (int): Number of results to return
            namespace (str, optional): Namespace to query
            
        Returns:
            list: List of (id, score, metadata) tuples
        """
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            return [(match.id, match.score, match.metadata) for match in results.matches]
            
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise
    
    def delete_documents(self, ids, namespace=None):
        """Delete documents from the vector store.
        
        Args:
            ids (list): List of document IDs to delete
            namespace (str, optional): Namespace to delete from
        """
        try:
            self.index.delete(
                ids=ids,
                namespace=namespace
            )
            logger.info(f"Successfully deleted {len(ids)} documents from namespace: {namespace}")
            
        except Exception as e:
            logger.error(f"Error deleting documents from vector store: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Query text
            top_k: Number of results to return
            threshold: Minimum similarity score threshold
            filter: Optional metadata filter
            namespace: Optional namespace to search in
            
        Returns:
            List of document dictionaries with text, metadata, and score
        """
        try:
            # Use provided namespace or default
            namespace = namespace or self.namespace
            
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Search in Pinecone
            results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False,
                filter=filter
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                if match.score >= threshold:
                    formatted_results.append({
                        "id": match.id,
                        "text": match.metadata.get("text", ""),
                        "metadata": {k: v for k, v in match.metadata.items() if k != "text"},
                        "score": match.score
                    })
            
            logger.info(f"Found {len(formatted_results)} similar documents in namespace '{namespace}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def get_langchain_retriever(self, search_kwargs=None, namespace=None):
        """Get a LangChain retriever for the vector store.
        
        Args:
            search_kwargs: Optional search parameters
            namespace: Optional namespace to search in
            
        Returns:
            LangChain retriever object
        """
        # Use provided namespace or default
        namespace = namespace or self.namespace
        
        # Create a new PineconeVectorStore with the specified namespace
        langchain_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embedding_service.get_langchain_embeddings(),
            namespace=namespace,
            text_key="text"
        )
        
        # Get retriever with search parameters
        return langchain_store.as_retriever(search_kwargs=search_kwargs or {})
    
    def get_llama_retriever(self, top_k=TOP_K, namespace=None):
        """Get a LlamaIndex retriever for the vector store.
        
        Args:
            top_k: Number of documents to retrieve
            namespace: Optional namespace to search in
            
        Returns:
            LlamaIndex retriever object
        """
        # Use provided namespace or default
        namespace = namespace or self.namespace
        
        # Create a new LlamaIndex vector store with the specified namespace
        llama_store = LlamaIndexPineconeStore(
            pinecone_index=self.index,
            namespace=namespace,
            embed_model=self.embedding_service.get_llama_embeddings()
        )
        
        # Create a retriever
        return VectorIndexRetriever(
            vector_store=llama_store,
            top_k=top_k,
            similarity_top_k=top_k
        )
