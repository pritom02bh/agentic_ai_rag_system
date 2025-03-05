# 🌐 PharmaSync: Intelligent Medical Supply Chain Optimization AI

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9+-brightgreen.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-success.svg)](https://github.com/yourusername/your-repo)

## 🚀 Project Overview

PharmaSync is a revolutionary Artificial Intelligence ecosystem designed to transform medical supply chain management through advanced Retrieval-Augmented Generation (RAG) technology. Our intelligent platform delivers unprecedented real-time insights, optimizing pharmaceutical logistics, inventory precision, and strategic decision-making with unparalleled accuracy and efficiency.

## ⚙️ System Architecture

![System Architecture](https://github.com/user-attachments/assets/c271d6ce-7e6e-4a47-9f61-63b7dce85f33)

## 🌟 Key Features

### 💬 Intelligent Chat Interface
- Context-aware conversation handling
- Modern, responsive user interface
- Real-time response generation
- Rich text and markdown support

### 🧠 Advanced RAG Capabilities
- Multi-namespace document retrieval
- Hybrid semantic search
- Intelligent query processing
- Conversation history integration

### 📊 Supply Chain Analytics
- Real-time inventory monitoring
- Transport logistics analysis
- Data-driven recommendations
- Predictive supply chain insights

## 🛠️ Technology Stack

| Domain | Technology |
|--------|------------|
| **Backend** | Flask |
| **Language Model** | GPT-4o Mini |
| **Vector Database** | Pinecone |
| **Embeddings** | text-embedding-ada-002 |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Database** | SQLite |

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
