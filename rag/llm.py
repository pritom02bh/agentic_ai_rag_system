import logging
from typing import List, Dict, Any, Optional, Union

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

from config import OPENAI_API_KEY, MODEL_NAME, RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class LLMService:
    """Service for generating responses using OpenAI's language models."""
    
    def __init__(self, api_key: str = OPENAI_API_KEY, model_name: str = MODEL_NAME):
        """Initialize the LLM service.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the language model to use
        """
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize OpenAI client for direct API access
        self.openai_client = OpenAI(api_key=self.api_key)
        
        # Initialize LangChain ChatOpenAI model - primary LLM provider
        self.langchain_chat = ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self.api_key,
            temperature=0.7
        )
        
        # Initialize LlamaIndex OpenAI model for compatibility
        self.llama_llm = LlamaIndexOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.7
        )
        
        logger.info(f"Initialized LLM service with model: {self.model_name}")
    
    def generate_response(self, query: str, context: List[Dict[str, Any]], system_prompt: str = RAG_SYSTEM_PROMPT) -> str:
        """Generate a response using the LLM with context from retrieved documents."""
        try:
            # Format context from retrieved documents
            formatted_context = self._format_context(context)
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "Context information: {context}\n\nQuestion: {query}")
            ])
            
            # Create chain
            chain = prompt | self.langchain_chat | StrOutputParser()
            
            # Generate response
            response = chain.invoke({"context": formatted_context, "query": query})
            
            logger.info("Generated response using LangChain")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Fallback to direct OpenAI API
            return self._generate_response_direct(query, context, system_prompt)
    
    def _generate_response_direct(self, query: str, context: List[Dict[str, Any]], system_prompt: str) -> str:
        """Generate a response using the OpenAI API directly (fallback method)."""
        try:
            # Format context
            formatted_context = self._format_context(context)
            
            # Create messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context information: {formatted_context}\n\nQuestion: {query}"}
            ]
            
            # Generate completion
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            logger.info("Generated response using direct OpenAI API (fallback)")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in fallback response generation: {str(e)}")
            return "I'm sorry, I encountered an error while generating a response. Please try again."
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context from retrieved documents for inclusion in prompts.
        
        Args:
            context: List of retrieved documents with text and metadata
            
        Returns:
            Formatted context string
        """
        if not context:
            return "No relevant information found."
            
        formatted_docs = []
        for i, doc in enumerate(context):
            doc_text = doc["text"]
            doc_metadata = doc.get("metadata", {})
            source = doc_metadata.get("source", f"Document {i+1}")
            formatted_docs.append(f"[{source}]\n{doc_text}\n")
            
        return "\n\n".join(formatted_docs)
    
    def get_llama_llm(self):
        """Get the LlamaIndex LLM model.
        
        Returns:
            LlamaIndex LLM model
        """
        return self.llama_llm
