# 🚀🤖🌐🏥💊 PharmaSync: Multi-Agent AI-Powered RAG Analytics for Smart Medical Supply Chains

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-success.svg)](https://github.com/yourusername/your-repo)

## 🛠 Technology Stack

<p align="center">
  <a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://flask.palletsprojects.com/" target="_blank">
    <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  </a>
  <a href="https://openai.com/" target="_blank">
    <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
  </a>
  <a href="https://www.pinecone.io/" target="_blank">
    <img src="https://img.shields.io/badge/Pinecone-6E42BA?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSJ3aGl0ZSI+PHBhdGggZD0iTTEyIDJMMiA3bDEwIDUgMTAtNS0xMC01em0wIDEybDEwLTUgMTAgNS0xMCA1LTEwLTV6bTAgMTJsMTAtNSAxMCA1LTEwIDUtMTAtNXoiLz48L3N2Zz4=" alt="Pinecone">
  </a>
</p>

## 🖼️ Project Visualizations

<p align="center">
  <img src="https://via.placeholder.com/800x400.png?text=PharmaSync+Architecture+Diagram" alt="System Architecture Diagram" width="800">
  
  <img src="https://via.placeholder.com/800x400.png?text=Supply+Chain+Data+Flow+Visualization" alt="Data Flow Visualization" width="800">
  
  <img src="https://via.placeholder.com/800x400.png?text=AI+Powered+Logistics+Insights" alt="AI Insights Visualization" width="800">
</p>

## 🚀 Project Overview

PharmaSync is a revolutionary Artificial Intelligence ecosystem designed to transform medical supply chain management through advanced Retrieval-Augmented Generation (RAG) technology. Our intelligent platform delivers unprecedented real-time insights, optimizing pharmaceutical logistics, inventory precision, and strategic decision-making with unparalleled accuracy and efficiency.

## ⚙️ System Architecture

![System Architecture](https://github.com/user-attachments/assets/c271d6ce-7e6e-4a47-9f61-63b7dce85f33)

## 🌟 Key Features

### 🤖 Intelligent Multi-Agent System
- **Router Agent** 🚦: Orchestrates workflow and query processing
  - Intelligent query analysis
  - Dynamic agent coordination
  - Contextual understanding

- **Web Search Agent** 🌐: Real-time market intelligence
  - Gather current logistics data
  - Track market trends
  - Update knowledge base dynamically

- **Analytical Expert Agent** 📊: Advanced data analysis
  - Deep dive into supply chain metrics
  - Predictive modeling
  - Complex pattern recognition

- **Report Generation Agent** 📝: Comprehensive insights generation
  - Synthesize cross-agent findings
  - Create structured, actionable reports
  - Provide strategic recommendations

### 🧠 Advanced RAG Capabilities
- Multi-namespace document retrieval
- Hybrid semantic search
- Intelligent query processing
- Conversation history integration
- Contextual learning and adaptation

### 📊 Supply Chain Optimization
- Real-time inventory monitoring
- Predictive logistics planning
- Risk assessment and mitigation
- Performance optimization recommendations
- Strategic decision support

## 🛠️ Technology Stack

| Domain | Technology | Details |
|--------|------------|---------|
| **Backend Framework** | 🌐 Flask | Python-based web framework |
| **Language Model** | 🤖 OpenAI GPT-4o Mini | Advanced AI processing |
| **Vector Database** | 🗃️ Pinecone | Semantic search and retrieval |
| **Embeddings** | 📊 text-embedding-ada-002 | Advanced semantic encoding |
| **Frontend** | 💻 HTML5, CSS3, JavaScript | Modern web technologies |
| **Database** | 🗄️ SQLite | Lightweight, serverless database |

### 🔗 Key Technology Integrations

<p align="center">
  <img src="https://via.placeholder.com/1000x300.png?text=Technology+Integration+Ecosystem" alt="Technology Integration" width="1000">
</p>

#### Core Technology Interactions
- **AI-Powered Retrieval**: OpenAI models enhancing search capabilities
- **Semantic Indexing**: Pinecone vector database for intelligent data mapping
- **Scalable Backend**: Flask providing robust web application framework
- **Efficient Data Handling**: SQLite for lightweight, fast data storage

## 🔄 Workflow Highlights

1. **Query Submission**
   - Users submit supply chain or logistics questions
   - System captures contextual conversation history

2. **Intelligent Processing**
   - Advanced routing determines query intent
   - Multi-source information retrieval
   - Cross-reference and knowledge integration

3. **Intelligent Response**
   - Structured, actionable insights
   - Supporting data and recommendations
   - Source-attributed information

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Pinecone API Key
- OpenAI API Key

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd medical-supply-chain-rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Environment Variables:
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_index
   PINECONE_DIMENSION=1536
   PINECONE_METRIC=cosine
   ```

### Running the Application
```bash
python run_ui.py
```
Access the UI at `http://localhost:5000`

## 📊 Use Cases

- **Inventory Optimization** 📦
  - Identify optimal safety stock levels
  - Predict inventory requirements

- **Logistics Planning** 🚚
  - Analyze carrier performance
  - Optimize transportation routes

- **Supply Chain Risk Management** ⚠️
  - Identify potential disruption risks
  - Recommend alternative suppliers

## 🔍 Multi-Namespace RAG System

### Namespaces
1. **Inventory Namespace** 
   - Product details
   - Stock levels
   - Pricing information
   - Supplier details

2. **Transport Namespace**
   - Shipment tracking
   - Logistics performance
   - Carrier metrics

### Query Types
- Inventory-only queries
- Transport-only queries
- Multi-namespace integrated queries

## 🛡️ Security Features
- Environment variable protection
- Secure API key management
- Input sanitization
- Rate limiting implementation

## 📝 Configuration Parameters
- `CHUNK_SIZE`: 512
- `CHUNK_OVERLAP`: 50
- `TOP_K`: 5
- `SIMILARITY_THRESHOLD`: 0.7

## 👥 Author
**Pritom Bhowmik**

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

**Empowering Medical Supply Chains with Intelligent AI** 🚀🏥
