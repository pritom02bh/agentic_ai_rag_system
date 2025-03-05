import logging
from typing import Dict, Any, List, Optional
import requests
import json
import datetime

from rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

class AgentTools:
    """Collection of tools that can be used by agents in the RAG system."""
    
    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        """Initialize the agent tools.
        
        Args:
            rag_pipeline: RAG pipeline for document retrieval and querying
        """
        self.rag_pipeline = rag_pipeline or RAGPipeline()
    
    def search_documents(self, query: str) -> str:
        """Search for information in the document collection.
        
        Args:
            query: Search query
            
        Returns:
            Formatted search results as a string
        """
        try:
            results = self.rag_pipeline.query(query)
            
            # Format the results as a string
            response = f"Answer: {results['answer']}\n\nSources:\n"
            
            for i, source in enumerate(results['sources']):
                source_name = source['metadata'].get('source', 'Unknown')
                response += f"[{i+1}] {source_name}: {source['text']}\n"
            
            return response
        
        except Exception as e:
            logger.error(f"Error in search_documents: {str(e)}")
            return f"Error searching documents: {str(e)}"
    
    def get_current_time(self, _: str = "") -> str:
        """Get the current date and time.
        
        Args:
            _: Ignored parameter to maintain tool interface
            
        Returns:
            Current date and time as a string
        """
        now = datetime.datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    def web_search(self, query: str) -> str:
        """Perform a web search for information.
        
        Note: This is a mock implementation. In a production environment,
        you would integrate with a real search API like Google or Bing.
        
        Args:
            query: Search query
            
        Returns:
            Search results as a string
        """
        # This is a mock implementation
        return f"Web search results for '{query}' would appear here. In a real implementation, this would connect to a search API."
    
    def calculate(self, expression: str) -> str:
        """Evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of the calculation as a string
        """
        try:
            # Use eval with restricted globals for basic calculations
            # Note: This is for demonstration purposes only. In production,
            # you should use a safer approach to evaluate expressions.
            result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "max": max, "min": min})
            return f"Result: {result}"
        
        except Exception as e:
            logger.error(f"Error in calculate: {str(e)}")
            return f"Error calculating expression: {str(e)}"
    
    def get_weather(self, location: str) -> str:
        """Get current weather information for a location.
        
        Note: This is a mock implementation. In a production environment,
        you would integrate with a weather API.
        
        Args:
            location: Location to get weather for
            
        Returns:
            Weather information as a string
        """
        # This is a mock implementation
        return f"Weather information for '{location}' would appear here. In a real implementation, this would connect to a weather API."
