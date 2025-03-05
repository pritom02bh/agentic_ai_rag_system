import logging
from typing import List, Dict, Any, Optional, Callable, Union

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory
import re

from config import AGENT_SYSTEM_PROMPT, AGENT_MAX_ITERATIONS
from rag.llm import LLMService

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for implementing agents in the RAG system."""
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        max_iterations: int = AGENT_MAX_ITERATIONS
    ):
        """Initialize the base agent.
        
        Args:
            llm_service: Service for LLM interactions
            system_prompt: System prompt for the agent
            max_iterations: Maximum number of iterations for the agent
        """
        self.llm_service = llm_service or LLMService()
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.tools = []
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Agent components will be initialized in setup_agent
        self.prompt = None
        self.output_parser = None
        self.llm_chain = None
        self.agent = None
        self.agent_executor = None
        
        logger.info("Initialized base agent")
    
    def add_tool(self, name: str, func: Callable, description: str):
        """Add a tool to the agent.
        
        Args:
            name: Name of the tool
            func: Function to call when the tool is used
            description: Description of what the tool does
        """
        tool = Tool(
            name=name,
            func=func,
            description=description
        )
        self.tools.append(tool)
        logger.info(f"Added tool '{name}' to agent")
    
    def setup_agent(self):
        """Set up the agent with tools and LLM."""
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement setup_agent")
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the agent on a query.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with the agent's response
        """
        # Ensure agent is set up
        if not self.agent_executor:
            self.setup_agent()
        
        try:
            # Run the agent
            result = self.agent_executor.run(input=query)
            
            return {
                "query": query,
                "answer": result,
                "agent_type": self.__class__.__name__
            }
        
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            # Return a graceful error message
            return {
                "query": query,
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "agent_type": self.__class__.__name__,
                "error": str(e)
            }


class CustomPromptTemplate(StringPromptTemplate):
    """Custom prompt template for the agent."""
    
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        intermediate_steps = kwargs.pop("intermediate_steps")
        
        # Format the observations as a string
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
        
        # Set the agent_scratchpad variable to the formatted thoughts
        kwargs["agent_scratchpad"] = thoughts
        
        # Create a list of tool names and descriptions for the prompt
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tools"] = tools_str
        
        # Create a list of tool names for the prompt
        tool_names = ", ".join([tool.name for tool in self.tools])
        kwargs["tool_names"] = tool_names
        
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    """Custom output parser for the agent."""
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # Check if the agent wants to finish
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )
        
        # Parse the action and input
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text, re.DOTALL)
        
        if not match:
            raise ValueError(f"Could not parse LLM output: {text}")
        
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        
        return AgentAction(tool=action, tool_input=action_input, log=text)
