import logging
from typing import List, Dict, Any, Optional

from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import LLMChain

from config import AGENT_SYSTEM_PROMPT, AGENT_MAX_ITERATIONS
from rag.llm import LLMService
from rag.rag_pipeline import RAGPipeline
from .base_agent import BaseAgent, CustomPromptTemplate, CustomOutputParser
from .tools import AgentTools

logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory for creating different types of agents."""
    
    @staticmethod
    def create_agent(agent_type: str, rag_pipeline: Optional[RAGPipeline] = None) -> BaseAgent:
        """Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            rag_pipeline: RAG pipeline for document operations
            
        Returns:
            Initialized agent instance
            
        Raises:
            ValueError: If the agent type is not supported
        """
        if agent_type.lower() == "research":
            return ResearchAgent(rag_pipeline=rag_pipeline)
        elif agent_type.lower() == "qa":
            return QAAgent(rag_pipeline=rag_pipeline)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")


class ResearchAgent(BaseAgent):
    """Agent specialized for research tasks."""
    
    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        llm_service: Optional[LLMService] = None,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        max_iterations: int = AGENT_MAX_ITERATIONS
    ):
        """Initialize the research agent.
        
        Args:
            rag_pipeline: RAG pipeline for document operations
            llm_service: Service for LLM interactions
            system_prompt: System prompt for the agent
            max_iterations: Maximum number of iterations for the agent
        """
        super().__init__(llm_service, system_prompt, max_iterations)
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.tools_manager = AgentTools(rag_pipeline=self.rag_pipeline)
        
        # Add tools
        self.add_tool(
            "search_documents",
            self.tools_manager.search_documents,
            "Search for information in the document collection. Input should be a specific question."
        )
        self.add_tool(
            "web_search",
            self.tools_manager.web_search,
            "Search the web for information not found in the documents. Input should be a search query."
        )
        self.add_tool(
            "get_current_time",
            self.tools_manager.get_current_time,
            "Get the current date and time. No input required."
        )
        
        logger.info("Initialized research agent with tools")
    
    def setup_agent(self):
        """Set up the research agent with tools and LLM."""
        # Define the prompt template
        template = """
        You are a research assistant that helps users find information.
        
        {system_prompt}
        
        You have access to the following tools:
        
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Question: {input}
        {agent_scratchpad}
        """
        
        # Create the prompt template
        self.prompt = CustomPromptTemplate(
            template=template,
            tools=self.tools,
            input_variables=["input", "intermediate_steps", "system_prompt"]
        )
        
        # Create the output parser
        self.output_parser = CustomOutputParser()
        
        # Create the LLM chain
        self.llm_chain = LLMChain(
            llm=self.llm_service.get_langchain_chat_model(),
            prompt=self.prompt
        )
        
        # Create the agent
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=self.max_iterations,
            memory=self.memory
        )
        
        logger.info("Research agent setup complete")


class QAAgent(BaseAgent):
    """Agent specialized for question answering tasks."""
    
    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        llm_service: Optional[LLMService] = None,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        max_iterations: int = AGENT_MAX_ITERATIONS
    ):
        """Initialize the QA agent.
        
        Args:
            rag_pipeline: RAG pipeline for document operations
            llm_service: Service for LLM interactions
            system_prompt: System prompt for the agent
            max_iterations: Maximum number of iterations for the agent
        """
        super().__init__(llm_service, system_prompt, max_iterations)
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.tools_manager = AgentTools(rag_pipeline=self.rag_pipeline)
        
        # Add tools
        self.add_tool(
            "search_documents",
            self.tools_manager.search_documents,
            "Search for information in the document collection. Input should be a specific question."
        )
        self.add_tool(
            "calculate",
            self.tools_manager.calculate,
            "Perform a calculation. Input should be a mathematical expression."
        )
        
        logger.info("Initialized QA agent with tools")
    
    def setup_agent(self):
        """Set up the QA agent with tools and LLM."""
        # Define the prompt template
        template = """
        You are a question answering assistant that helps users find accurate information.
        
        {system_prompt}
        
        You have access to the following tools:
        
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Always cite your sources when providing information from the documents.
        
        Begin!
        
        Question: {input}
        {agent_scratchpad}
        """
        
        # Create the prompt template
        self.prompt = CustomPromptTemplate(
            template=template,
            tools=self.tools,
            input_variables=["input", "intermediate_steps", "system_prompt"]
        )
        
        # Create the output parser
        self.output_parser = CustomOutputParser()
        
        # Create the LLM chain
        self.llm_chain = LLMChain(
            llm=self.llm_service.get_langchain_chat_model(),
            prompt=self.prompt
        )
        
        # Create the agent
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=self.max_iterations,
            memory=self.memory
        )
        
        logger.info("QA agent setup complete")
