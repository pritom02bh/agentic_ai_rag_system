# 📦 Agentic AI RAG for Pharmaceutical Supply Chain & Logistics 🚚

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9+-brightgreen.svg)](https://www.python.org/downloads/)
[![Agenta Framework](https://img.shields.io/badge/Framework-Agenta-orange.svg)](https://github.com/your-agenta-repo)
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-success.svg)](https://github.com/yourusername/your-repo)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 🚀 Overview

This project implements an **Agentic AI Retrieval-Augmented Generation (RAG)** system to streamline and enhance decision-making in the **Supply Chain and Logistics** domain. Leveraging the power of large language models (specifically **Llama3-70B**), intelligent agents, and a robust vector database, this system provides users with accurate, up-to-date, and insightful information through a user-friendly chatbot interface.

## ⚙️ System Architecture
<img src="https://github.com/user-attachments/assets/c271d6ce-7e6e-4a47-9f61-63b7dce85f33" width="75%">

## 🗄️ File Structure
![image](https://github.com/user-attachments/assets/8059b9df-38d0-4147-a358-685493c26644)

## 🌟 Key Features

- **Intuitive Chatbot Interface** 💬 
  - Natural language query processing
  - Contextual conversation history
  - Rich response formatting with tables and charts

- **Intelligent Multi-Agent System** 🧠
  - **Router Agent** 🚦 - Analyzes queries and orchestrates workflow
  - **Web Search Agent** 🌐 - Gathers real-time market data and logistics news
  - **Analytical Expert Agent** 📊 - Performs in-depth data analysis
  - **Report Generation Agent** 📝 - Creates comprehensive summaries

- **Advanced RAG Capabilities** 📚
  - Semantic search across logistics documents
  - Hybrid retrieval combining keyword and vector search
  - Custom chunking for complex logistics documentation
  - **Multi-namespace Retrieval** 🔄 - Query across both inventory and transport data simultaneously

- **Enterprise-Grade Data Integration** 🔄
  - Transportation management system (TMS) integration
  - Inventory management system connectivity
  - Real-time freight rate lookups
  - Warehouse capacity utilization tracking

- **Supply Chain Analytics** 📈
  - Demand forecasting assistance
  - Route optimization suggestions
  - Inventory level recommendations
  - Risk assessment for supply disruptions

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **LLM Backend** | Llama3-70B (via Groq) |
| **Framework** | Agenta |
| **Web Search** | DuckDuckGo API, Newspaper4k |
| **Vector Database** | Pinecone |
| **Embeddings** | text-embedding-ada-002 |
| **Programming** | Python 3.9+ |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |

## 🔄 Workflow Explanation

1. **Query Submission** 🔍
   - User submits logistics or supply chain question through chat interface
   - System captures context from conversation history

2. **Intelligent Routing** 🧭
   - Router Agent analyzes query intent and complexity
   - Determines optimal processing path and required agents

3. **Multi-Source Information Retrieval** 📊
   - **Web Search**: Real-time freight rates, fuel prices, port delays
   - **Analytics**: Historical performance data, seasonal patterns
   - **RAG**: Company-specific logistics policies, carrier guidelines
   - **Multi-namespace Retrieval**: Combines data from inventory and transport namespaces

4. **Knowledge Integration** 🧩
   - Cross-reference information from multiple sources
   - Resolve conflicts between data points
   - Prioritize information based on relevance and recency

5. **Report Generation** 📄
   - Structured response with executive summary
   - Supporting data, charts, and actionable recommendations
   - Citations to source information

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Pinecone API key
- OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd medical-supply-chain-rag
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file with your API keys:
     ```
     OPENAI_API_KEY=your_openai_api_key
     PINECONE_API_KEY=your_pinecone_api_key
     ```

### Running the Application

1. Start the web UI:
   ```
   python run_ui.py
   ```

2. Access the UI at http://localhost:5000

## 📊 Use Cases

- **Inventory Optimization** 📦
  - "What's the optimal safety stock level for product XYZ based on current lead times?"
  - "How will the port congestion in Shanghai affect our Q3 inventory planning?"

- **Transportation Planning** 🚛
  - "What carriers have the best on-time delivery rate for the Midwest region?"
  - "Compare costs between intermodal and full truckload for our West Coast shipments"

- **Supply Chain Risk Management** ⚠️
  - "Identify potential disruption risks for our semiconductor components"
  - "What backup suppliers do we have for raw materials in Southeast Asia?"

- **Performance Analytics** 📈
  - "Generate a report on warehouse efficiency trends over the past 12 months"
  - "Compare carrier performance metrics across our top 5 logistics partners"

## 🔍 Multi-namespace RAG System

Our system is designed to retrieve and generate information from two distinct namespaces:

1. **Inventory Namespace** 📦: Contains information about medical products, including:
   - Product IDs and names
   - Current stock levels
   - Pricing information
   - Expiration dates
   - Storage requirements
   - Supplier details

2. **Transport Namespace** 🚚: Contains information about shipments and logistics, including:
   - Recent shipments (IDs, dates, contents)
   - Shipping performance metrics
   - Carrier performance
   - Route optimization

### Querying the System

The system supports three types of queries:

1. **Inventory-only Queries**: Retrieve information only from the inventory namespace.
   - Example: "What is the current stock level of surgical masks?"

2. **Transport-only Queries**: Retrieve information only from the transport namespace.
   - Example: "When was the last shipment of surgical masks delivered?"

3. **Multi-namespace Queries**: Retrieve and combine information from both namespaces.
   - Example: "What is the current stock level of surgical masks and when was the last shipment delivered?"

### API Endpoints

- `POST /query`: Submit a query to the RAG system
  - Parameters:
    - `query` (string, required): The query text
    - `namespace` (string, optional): Specify "inventory" or "transport" to query a specific namespace. If not provided, the system will query both namespaces.

  Example request:
  ```json
  {
    "query": "What is the current stock level of surgical masks and when was the last shipment delivered?"
  }
  ```

### Testing

To test the multi-namespace functionality, run:

```
python test_rag_query.py
```

This script will:
1. Query the inventory namespace
2. Query the transport namespace
3. Perform a multi-namespace query

<p align="left">
  <b>Developed by Pritom Bhowmik</b>
</p>
