# 📦 Agentic AI RAG for Supply Chain & Logistics 🚚

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9+-brightgreen.svg)](https://www.python.org/downloads/)
[![Agenta Framework](https://img.shields.io/badge/Framework-Agenta-orange.svg)](https://github.com/your-agenta-repo)
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-success.svg)](https://github.com/yourusername/your-repo)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 🚀 Overview

This project implements an **Agentic AI Retrieval-Augmented Generation (RAG)** system to streamline and enhance decision-making in the **Supply Chain and Logistics** domain. Leveraging the power of large language models (specifically **Llama3-70B**), intelligent agents, and a robust vector database, this system provides users with accurate, up-to-date, and insightful information through a user-friendly chatbot interface.

## 🌟 Features

* **Chatbot Interface:** 💬 Intuitive user interface for easy interaction and query input.
* **Intelligent Routing:** 🚦 A Router Agent analyzes user queries and directs them to the appropriate processing agents.
* **Web Search Capability:** 🌐 Utilizes DuckDuckGo and Newspaper4k tools to gather real-time information from the web.
* **Analytical Expertise:** 📊 An Analytical Expert Agent performs in-depth data analysis and generates actionable insights.
* **RAG System:** 📚 Retrieves relevant data from a vector database containing Transportation and Inventory information.
* **Data Aggregation:** 🔄 Combines data from various sources for comprehensive reporting.
* **Report Generation:** 📝 Formats and presents information in a clear and concise manner.
* **Powered by Agenta:** 🤖 Built on the Agenta framework for seamless agent orchestration.
* **Llama3-70B Model:** 🧠 Leverages the advanced capabilities of the Llama3-70B model for accurate and context-aware responses.

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **LLM Backend** | Llama3-70B (via Groq) |
| **Framework** | Agenta |
| **Web Search** | DuckDuckGo API, Newspaper4k |
| **Vector Database** | Chroma DB |
| **Embeddings** | E5-large-v2 |
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
- Groq API key (for Llama3-70B access)
- DuckDuckGo API credentials



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



<p align="left">
  <b>Developed by Pritom Bhowmik</b>
</p>
