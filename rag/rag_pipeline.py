import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document as LlamaDocument

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import TOP_K, RAG_SYSTEM_PROMPT
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .retriever import Retriever
from .llm import LLMService
from .output_formatter import OutputFormatter

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline that orchestrates document processing, retrieval, and generation."""
    
    def __init__(
        self,
        vector_store=None,
        embedding_service=None,
        document_processor=None,
        retriever=None,
        llm_service=None,
        namespace="inventory"  # Default namespace
    ):
        """Initialize the RAG pipeline with its components.
        
        Args:
            vector_store: Vector store for document storage and retrieval
            embedding_service: Service for generating embeddings
            document_processor: Service for processing documents
            retriever: Service for retrieving relevant documents
            llm_service: Large language model service
            namespace: Namespace to use (inventory or transport)
        """
        # Validate namespace
        if namespace not in ["inventory", "transport"]:
            logger.warning(f"Invalid namespace: {namespace}. Using 'inventory' as default.")
            namespace = "inventory"
            
        self.namespace = namespace
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore(
            embedding_service=self.embedding_service,
            namespace=self.namespace
        )
        self.document_processor = document_processor or DocumentProcessor()
        self.retriever = retriever or Retriever(
            vector_store=self.vector_store,
            namespace=self.namespace
        )
        self.llm_service = llm_service or LLMService()
        self.output_formatter = OutputFormatter()
        
        # Add conversation history
        self.conversation_history = []
        self.max_history_length = 5  # Keep last 5 interactions for context
        
        # LlamaIndex components
        self.llama_index = None
        self.llama_query_engine = None
        
        # LangChain components
        self.qa_chain = None
        
        logger.info(f"Initialized RAG pipeline with namespace '{self.namespace}'")
    
    def _add_to_history(self, query: str, response: str):
        """Add a query-response pair to conversation history."""
        self.conversation_history.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only the last N interactions
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def _get_conversation_context(self) -> str:
        """Format conversation history into context string."""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for idx, interaction in enumerate(self.conversation_history[-3:]):  # Only use last 3 interactions for relevance
            context_parts.append(f"Interaction {idx + 1}:\nUser: {interaction['query']}\nAssistant: {interaction['response']}\n")
        
        return "\n".join(context_parts)

    def _enhance_query_with_context(self, query_text: str) -> str:
        """Enhance the query with conversation context and reference resolution."""
        # Get recent conversation context
        conversation_context = self._get_conversation_context()
        
        # Extract potential references (IDs, codes, etc.) from the query
        query_lower = query_text.lower()
        reference_keywords = ['this', 'that', 'it', 'the shipment', 'the item']
        
        has_reference = any(keyword in query_lower for keyword in reference_keywords)
        
        if has_reference and self.conversation_history:
            # Look for specific IDs or codes in recent conversation
            last_response = self.conversation_history[-1]['response']
            augmented_query = f"""
Context from recent conversation:
{conversation_context}

Current question: {query_text}

Please consider any IDs, codes, or specific items mentioned in the conversation history when answering this question.
If the question refers to a specific item or shipment mentioned earlier, use that information in the response.
"""
        else:
            augmented_query = f"{conversation_context}\n\nCurrent question: {query_text}"
        
        return augmented_query

    def ingest_document(self, file_path: str, namespace: Optional[str] = None) -> List[str]:
        """Process and ingest a single document into the vector store.
        
        Args:
            file_path: Path to the document file
            namespace: Optional namespace to use (inventory or transport)
            
        Returns:
            List of IDs for the ingested document chunks
        """
        # Use provided namespace or default
        namespace = namespace or self.namespace
        
        try:
            # Process the document
            document_chunks = self.document_processor.process_document(file_path)
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents(document_chunks, namespace=namespace)
            
            logger.info(f"Ingested document '{file_path}' with {len(doc_ids)} chunks into namespace '{namespace}'")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            raise
    
    def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
        namespace: Optional[str] = None
    ) -> List[str]:
        """Process and ingest documents into the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
            namespace: Optional namespace to use (inventory or transport)
            
        Returns:
            List of IDs for the ingested documents
        """
        # Use provided namespace or default
        namespace = namespace or self.namespace
        
        # Validate namespace
        if namespace not in ["inventory", "transport"]:
            logger.warning(f"Invalid namespace: {namespace}. Using '{self.namespace}' as default.")
            namespace = self.namespace
            
        try:
            # Process documents
            processed_docs = self.document_processor.process_documents(documents)
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents(processed_docs, namespace=namespace)
            
            logger.info(f"Ingested {len(doc_ids)} documents into namespace '{namespace}'")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            raise
    
    def query(
        self,
        query_text: str,
        namespace: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        namespace = namespace or self.namespace
        
        if namespace not in ["inventory", "transport"]:
            logger.warning(f"Invalid namespace: {namespace}. Using '{self.namespace}' as default.")
            namespace = self.namespace
            
        try:
            # Check if this is a casual message that doesn't require vector search
            text = query_text.lower()
            is_casual = (
                len(text.split()) < 5 or
                any(greeting in text for greeting in ['hi', 'hello', 'hey', 'greetings']) or
                not any(keyword in text for keyword in ['inventory', 'transport', 'stock', 'shipment', 'supply', 'medical'])
            )
            
            if is_casual and not self.conversation_history:  # Only treat as casual if it's the first message
                # Use LLM directly for casual conversation
                casual_system_prompt = """You are a friendly and professional medical supply chain assistant. 
                For casual greetings or general questions, respond in a warm, helpful manner.
                Keep responses concise and natural."""
                
                response = self.llm_service.generate_response(
                    query_text,
                    context=[],
                    system_prompt=casual_system_prompt
                )
                
                # Add to conversation history
                self._add_to_history(query_text, response)
                
                return {
                    "query": query_text,
                    "response": response,
                    "relevant_documents": []
                }

            # Enhance query with conversation context and reference resolution
            augmented_query = self._enhance_query_with_context(query_text)
            
            # For supply chain queries, use RAG pipeline
            relevant_docs = self.retriever.retrieve(augmented_query, namespace=namespace)
            
            # Add conversation history to the context
            system_prompt = f"""You are a medical supply chain assistant. Consider the following conversation history and context when answering:

Previous conversation:
{self._get_conversation_context()}

Important Instructions:
1. If the user refers to a specific shipment, item, or ID mentioned in previous messages, use that information in your response.
2. If you find relevant information in the context about the referenced item, include it in your response.
3. If you can't find specific information about a referenced item, acknowledge the reference but state that you don't have the specific information.
4. Be explicit about which items or shipments you're referring to in your response.

Answer the current question based on both the conversation history and the provided context."""
            
            # Generate response using LLM with conversation context
            raw_response = self.llm_service.generate_response(
                query_text,
                context=relevant_docs,
                system_prompt=system_prompt
            )
            
            formatted_response = self.output_formatter.format_response(raw_response)
            
            # Add to conversation history
            self._add_to_history(query_text, formatted_response.summary)
            
            return {
                "query": query_text,
                "response": formatted_response.summary,
                "formatted_response": formatted_response.to_dict(),
                "relevant_documents": relevant_docs,
                "namespace": namespace
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def query_multi_namespace(
        self,
        query_text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a query across multiple namespaces and combine the results."""
        try:
            # Check if this is a casual message that doesn't require vector search
            text = query_text.lower()
            is_casual = (
                len(text.split()) < 5 or
                any(greeting in text for greeting in ['hi', 'hello', 'hey', 'greetings']) or
                not any(keyword in text for keyword in ['inventory', 'transport', 'stock', 'shipment', 'supply', 'medical'])
            )
            
            if is_casual:
                # Use LLM directly for casual conversation
                casual_system_prompt = """You are a friendly and professional medical supply chain assistant. 
                For casual greetings or general questions, respond in a warm, helpful manner.
                Keep responses concise and natural."""
                
                response = self.llm_service.generate_response(
                    query_text,
                    context=[],
                    system_prompt=casual_system_prompt
                )
                
                # Add to conversation history
                self._add_to_history(query_text, response)
                
                return {
                    "query": query_text,
                    "response": response,
                    "relevant_documents": [],
                    "inventory_documents": [],
                    "transport_documents": []
                }

            # Get conversation history context
            conversation_context = self._get_conversation_context()
            
            # For supply chain queries, retrieve from both namespaces
            # Include conversation context in the retrieval
            augmented_query = f"{conversation_context}\n\nCurrent question: {query_text}"
            
            inventory_docs = self.retriever.retrieve(augmented_query, namespace="inventory")
            transport_docs = self.retriever.retrieve(augmented_query, namespace="transport")
            
            # Combine documents and add namespace metadata
            combined_docs = []
            for doc in inventory_docs:
                doc.setdefault("metadata", {})["namespace"] = "inventory"
                combined_docs.append(doc)
            for doc in transport_docs:
                doc.setdefault("metadata", {})["namespace"] = "transport"
                combined_docs.append(doc)
            
            # Add conversation history to the context
            system_prompt = f"""You are a medical supply chain assistant. Consider the following conversation history and context when answering:

Previous conversation:
{conversation_context}

Answer the current question based on both the conversation history and the provided context. 
If referring to items or information from previous questions, be explicit about which items you're referring to."""
            
            # Generate response using LLM with conversation context
            response = self.llm_service.generate_response(
                query_text,
                context=combined_docs,
                system_prompt=system_prompt
            )
            
            # Add to conversation history
            self._add_to_history(query_text, response)
            
            return {
                "query": query_text,
                "response": response,
                "relevant_documents": combined_docs,
                "inventory_documents": inventory_docs,
                "transport_documents": transport_docs
            }
            
        except Exception as e:
            logger.error(f"Error processing multi-namespace query: {str(e)}")
            raise
    
    def initialize_llama_index(self, namespace: Optional[str] = None):
        """Initialize LlamaIndex components for alternative querying.
        
        Args:
            namespace: Optional namespace to use
        """
        try:
            # Use provided namespace or default
            namespace = namespace or self.namespace
            
            # Get the LlamaIndex vector store
            llama_vector_store = self.vector_store.get_llama_vector_store()
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=llama_vector_store)
            
            # Create the index
            self.llama_index = VectorStoreIndex.from_vector_store(
                vector_store=llama_vector_store,
                storage_context=storage_context,
            )
            
            # Create retriever
            llama_retriever = VectorIndexRetriever(
                index=self.llama_index,
                similarity_top_k=TOP_K,
            )
            
            # Set the retriever in our retriever component
            self.retriever.set_llama_retriever(llama_retriever)
            
            # Create query engine
            self.llama_query_engine = RetrieverQueryEngine.from_args(
                retriever=llama_retriever,
                llm=self.llm_service.get_llama_llm(),
            )
            
            logger.info(f"Initialized LlamaIndex components for namespace '{namespace}'")
        
        except Exception as e:
            logger.error(f"Error initializing LlamaIndex: {str(e)}")
            raise
    
    def initialize_langchain_qa(self, namespace: Optional[str] = None):
        """Initialize LangChain QA chain for alternative querying.
        
        Args:
            namespace: Optional namespace to use
        """
        try:
            # Use provided namespace or default
            namespace = namespace or self.namespace
            
            # Get LangChain retriever
            langchain_retriever = self.vector_store.get_langchain_retriever(namespace=namespace)
            
            # Create prompt template
            template = """You are an AI assistant providing accurate information based on the given context.
            
            Context: {context}
            
            Question: {question}
            
            Answer the question based on the context provided. If the answer is not in the context, say 'I don't have enough information to answer this question.' and don't make up an answer."""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm_service.get_langchain_llm(),
                chain_type="stuff",
                retriever=langchain_retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            
            logger.info(f"Initialized LangChain QA chain for namespace '{namespace}'")
        
        except Exception as e:
            logger.error(f"Error initializing LangChain QA: {str(e)}")
            raise
    
    def query_with_llama_index(self, query_text: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Query using LlamaIndex components.
        
        Args:
            query_text: User query text
            namespace: Optional namespace to search in
            
        Returns:
            Dictionary with query results
        """
        # Use provided namespace or default
        namespace = namespace or self.namespace
        
        if not self.llama_query_engine:
            self.initialize_llama_index(namespace=namespace)
        
        response = self.llama_query_engine.query(query_text)
        
        return {
            "query": query_text,
            "answer": str(response),
            "sources": [node.node.get_content() for node in response.source_nodes],
            "namespace": namespace
        }
    
    def query_with_langchain(self, query_text: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Query using LangChain QA chain.
        
        Args:
            query_text: User query text
            namespace: Optional namespace to search in
            
        Returns:
            Dictionary with query results
        """
        # Use provided namespace or default
        namespace = namespace or self.namespace
        
        if not self.qa_chain:
            self.initialize_langchain_qa(namespace=namespace)
        
        result = self.qa_chain({"query": query_text})
        
        return {
            "query": query_text,
            "answer": result["result"],
            "sources": result.get("source_documents", []),
            "namespace": namespace
        }

    def clear_chat_history(self):
        """Clear all conversation history permanently."""
        self.conversation_history = []
        logger.info("Chat history cleared successfully")
        return {"status": "success", "message": "Chat history cleared successfully"}

    def delete_chat_entry(self, timestamp: str):
        """Delete a specific chat entry by its timestamp.
        
        Args:
            timestamp: The timestamp of the chat entry to delete
            
        Returns:
            Dict with status and message
        """
        try:
            # Find and remove the chat entry with matching timestamp
            self.conversation_history = [
                entry for entry in self.conversation_history 
                if entry["timestamp"] != timestamp
            ]
            logger.info(f"Chat entry with timestamp {timestamp} deleted successfully")
            return {
                "status": "success",
                "message": f"Chat entry deleted successfully"
            }
        except Exception as e:
            logger.error(f"Error deleting chat entry: {str(e)}")
            return {
                "status": "error",
                "message": f"Error deleting chat entry: {str(e)}"
            }
