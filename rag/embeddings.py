import logging
from typing import List, Dict, Any, Optional, Union

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding

from config import OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using OpenAI's embedding models."""
    
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = EMBEDDING_MODEL):
        """Initialize the embedding service.
        
        Args:
            api_key: OpenAI API key
            model: Name of the embedding model to use
        """
        self.api_key = api_key
        self.model = model
        
        # Initialize OpenAI client for direct API access
        self.openai_client = OpenAI(api_key=self.api_key)
        
        # Initialize LangChain embeddings
        self.langchain_embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=self.api_key
        )
        
        # Initialize LlamaIndex embeddings
        self.llama_embeddings = OpenAIEmbedding(
            model_name=self.model,
            api_key=self.api_key
        )
        
        logger.info(f"Initialized embedding service with model: {self.model}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text using OpenAI API directly.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as a list of floats
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI API directly.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_langchain_embeddings(self):
        """Get the LangChain embeddings model.
        
        Returns:
            LangChain embeddings model
        """
        return self.langchain_embeddings
    
    def get_llama_embeddings(self):
        """Get the LlamaIndex embeddings model.
        
        Returns:
            LlamaIndex embeddings model
        """
        return self.llama_embeddings
