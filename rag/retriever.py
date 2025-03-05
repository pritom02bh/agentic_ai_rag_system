import logging
from typing import List, Dict, Any, Optional, Union

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LangchainDocument
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

from config import TOP_K, SIMILARITY_THRESHOLD
from .vector_store import VectorStore
from .llm import LLMService

logger = logging.getLogger(__name__)

class Retriever:
    """Retriever for finding relevant documents based on a query."""
    
    def __init__(
        self,
        vector_store=None,
        llm_service=None,
        namespace="inventory"  # Default namespace
    ):
        """Initialize the retriever.
        
        Args:
            vector_store: Vector store for document retrieval
            llm_service: LLM service for reranking and response generation
            namespace: Default namespace to use (inventory or transport)
        """
        # Validate namespace
        if namespace not in ["inventory", "transport"]:
            logger.warning(f"Invalid namespace: {namespace}. Using 'inventory' as default.")
            namespace = "inventory"
            
        self.namespace = namespace
        self.vector_store = vector_store or VectorStore(namespace=self.namespace)
        self.llm_service = llm_service or LLMService()
        
        # Initialize LangChain retriever
        self.langchain_retriever = self.get_langchain_retriever(
            search_kwargs={"k": TOP_K},
            namespace=self.namespace
        )
        
        logger.info(f"Initialized retriever for namespace '{self.namespace}'")
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            threshold: Minimum similarity score threshold
            filter: Optional metadata filter
            namespace: Optional namespace to search in (inventory or transport)
            
        Returns:
            List of relevant document dictionaries
        """
        # Use provided namespace or default
        namespace = namespace or self.namespace
        
        # Validate namespace
        if namespace not in ["inventory", "transport"]:
            logger.warning(f"Invalid namespace: {namespace}. Using '{self.namespace}' as default.")
            namespace = self.namespace
            
        try:
            # Retrieve documents from vector store
            results = self.vector_store.similarity_search(
                query=query,
                top_k=top_k,
                threshold=threshold,
                filter=filter,
                namespace=namespace
            )
            
            logger.info(f"Retrieved {len(results)} documents from namespace '{namespace}'")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def get_langchain_retriever(
        self,
        search_kwargs: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ):
        """Get a LangChain retriever for the vector store.
        
        Args:
            search_kwargs: Optional search parameters
            namespace: Optional namespace to search in
            
        Returns:
            LangChain retriever object
        """
        # Use provided namespace or default
        namespace = namespace or self.namespace
        
        # Get retriever from vector store with namespace
        return self.vector_store.get_langchain_retriever(
            search_kwargs=search_kwargs,
            namespace=namespace
        )
    
    def get_llama_retriever(self, top_k: int = TOP_K, namespace: Optional[str] = None):
        """Get a LlamaIndex retriever for the vector store.
        
        Args:
            top_k: Number of documents to retrieve
            namespace: Optional namespace to search in (inventory or transport)
            
        Returns:
            LlamaIndex retriever object
        """
        # Use provided namespace or default
        namespace = namespace or self.namespace
        
        # Validate namespace
        if namespace not in ["inventory", "transport"]:
            logger.warning(f"Invalid namespace: {namespace}. Using '{self.namespace}' as default.")
            namespace = self.namespace
        
        # Create a LlamaIndex retriever from the vector store with namespace
        return self.vector_store.get_llama_retriever(top_k=top_k, namespace=namespace)
