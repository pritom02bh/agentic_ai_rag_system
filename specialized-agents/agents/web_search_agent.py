from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from datetime import datetime
import openai
import requests
import re
import os
from dotenv import load_dotenv

# Retrieve API keys
groq_api_key = os.getenv('GROQ_API_KEY')
phi_api_key = os.getenv('PHI_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

# user query
user_query = "Analyze the current updates on Chittagong Port including operational trends, policy changes, and logistical disruptions."

# Function to extract port details from the query using regex.
def extract_port(query: str) -> str:
    # The regex pattern here is simplistic and assumes that port names follow patterns like "Port of ..."
    pattern = r"(Port of [A-Za-z\s&-]+)"
    match = re.search(pattern, query)
    if match:
        return match.group(1).strip()
    return "Port of New York and New Jersey"  # Default port if none detected

# Extract the target port from the query
target_port = extract_port(user_query)

# Define the dynamic research agent for supply chain operations at any port
research_agent = Agent(
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGoTools(), Newspaper4kTools()],
    description=dedent(f"""\
        You are a Supply Chain Analyst specialized in monitoring real-time operational updates and trends 
        at high-traffic ports globally. Your current focus is on analyzing operations at {target_port}.
        Your expertise includes:

        - Deep investigative research on supply chain operations,
        - Fact-checking and source verification,
        - Data-driven reporting on current logistics trends,
        - Expert analysis of port operations impacting supply chains,
        - Trend analysis and implications for near-future operations,
        - Simplification of complex logistics and operational concepts,
        - Ensuring balanced and ethical perspectives,
        - Integrating global context to supply chain dynamics.
    """),
    instructions=dedent(f"""\
        1. Research Phase
           - Conduct web searches and gather the latest news and updates on port operations globally, then narrow down to {target_port}.
           - Prioritize real-time, authoritative, and non-historic sources.
           - Focus on current logistical challenges, operational disruptions, and policy changes that directly impact supply chain efficiency at {target_port}.

        2. Analysis Phase
           - Extract critical data points regarding current operations, policy adjustments, and logistical updates for {target_port}.
           - Analyze trends in real-time supply chain disruptions or operational improvements.
           - Assess the immediate impact of geopolitical factors and market conditions on port throughput and overall logistics flow.

        3. Writing Phase
           - Develop a compelling headline that reflects the latest operational updates at {target_port}.
           - Structure the report with clear sections focused on current operations:
             - Executive Summary, Critical Updates, Impact Analysis, and Future Outlook.
           - Include relevant quotes, statistics, and near-term implications.

        4. Quality Control
           - Verify all facts with cross-reference from multiple reputable sources.
           - Emphasize current operational details without including extensive historical or unrelated background.
           - Provide clear context for all data and offer actionable insights on operational trends.
    """),
    expected_output=dedent(f"""\
        # {{Compelling Headline on Current Port Operations and Supply Chain Impact}}

        ## Executive Summary
        {{Concise overview of the latest findings regarding operations at {target_port}.}}

        ## Critical Updates
        {{Recent operational changes and logistical disruptions at {target_port} that directly impact supply chains.}}
        {{Breaking news on policy changes and immediate adjustments by port authorities.}}
        {{Statistical evidence on port throughput and current operational trends.}}

        ## Impact Analysis
        {{Immediate implications on supply chain operations and logistics.}}
        {{Stakeholder perspectives including port authorities and key industry players.}}
        {{Analysis of market, geopolitical, and economic influences.}}

        ## Future Outlook
        {{Emerging trends and upcoming changes in port operations at {target_port}.}}
        {{Expert predictions on near-future performance and policy modifications.}}
        {{Potential challenges and actionable opportunities for supply chain optimization.}}

        ## Expert Insights
        {{Notable quotes and analysis from port operation experts and supply chain analysts.}}
        {{Discussion on contrasting viewpoints regarding current operational shifts.}}

        ## Sources & Methodology
        {{List of primary sources with key contributions.}}
        {{Overview of research methodology and data verification for current updates.}}
        
        ---
        Research conducted by Supply Chain Analyst  
        Logistics Analysis Report  
        Published: {{current_date}}  
        Last Updated: {{current_time}}
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

if __name__ == "__main__":
    research_agent.print_response(user_query, stream=True)
