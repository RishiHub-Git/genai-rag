# GenAI RAG Application

The **GenAI RAG Application** is a Python-based Retrieval-Augmented Generation (RAG) tool designed to provide precise answers to API-related guidelines. It uses vectorized document embeddings for context-based retrieval and a generative model for accurate responses. The application is hosted on Azure, leveraging its scalability and robustness.

---

## Features

- **Document Parsing**: Processes PDF files to extract and vectorize text.
- **Contextual Retrieval**: Retrieves relevant document chunks using vector similarity.
- **Generative AI**: Uses the Cohere LLM for precise and natural language answers.
- **REST API**: Exposes a `/api/generate` endpoint for easy integration with other tools.
- **Scalable**: Designed to run on Azure as a Web App or Function.

---

## Prerequisites

### Development Environment
- Python 3.9 or higher
- pip for dependency management

### Azure Setup
- Azure App Service or Function App
- Azure Storage for persisted data (if needed)
- Environment Variables:
  - `HuggingFaceKey`: HuggingFace API key for embeddings.
  - `cohere_api_key`: API key for Cohere LLM.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repository/genai-rag.git
   cd genai-rag

## Install Dependencies in VS Code
Activate your virtual environment:
$ python -m venv venv
$ venv\Scripts\activate

Upgrade pip to avoid version issues:
$ python -m pip install --upgrade pip


Install your project dependencies:
$ pip install -r requirements.txt