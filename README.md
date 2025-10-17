# AI-Chatbot

A simple AI-powered chatbot built in Python, designed to process and respond to user queries by combining language and media processing components.

<img width="1920" height="1200" alt="Screenshot (98)" src="https://github.com/user-attachments/assets/5ade0051-bdf9-4186-99dc-9279a3a6a9d6" />

## 🧩 Short Description

This project implements a modular chatbot application using Python. It supports embedding generation, vector databases, memory storage, media processing, and evaluation, and is deployable via a Streamlit interface.

## 🔍 Features

- **Embedding & Vector Search**: Generate embeddings and query a vector database for relevant content.  
- **Memory Storage**: Maintain conversational memory between sessions (via JSON-backed store).  
- **Media Processing**: Handle file inputs (e.g. images, audio) for richer interactions.  
- **Evaluation Module**: Evaluate response quality or correctness.  
- **Web App Interface**: Deploy via **Streamlit** (using `streamlit_app.py`).  
- **Configurable & Extensible**: Easily plug in different LLM or embedder backends.

## 📁 Repository Structure

.
├── .env # environment variables (e.g. API keys)
├── app.py # main app / entrypoint
├── embed.py # embedding & vector management logic
├── get_vector_db.py # module to initialize/manage vector DB
├── memory_store.py # persistent conversational memory
├── media_processing.py # image/audio/text processing utilities
├── query.py # query / prompt handling logic
├── evaluation.py # evaluation of generated responses
├── seed_data.py # initial seed data loader
├── streamlit_app.py # Streamlit frontend for chat interface
├── memory_store.json # persisted memory sensor
├── chat_memory.json # chat history store
├── requirements.txt # Python dependencies
└── README.md # this file

markdown
Copy code

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+  
- API keys / credentials (for embedding or language model services)  
- A vector database or embedding provider (FAISS, Pinecone, etc.)

### Installation

1. Clone the repo:  
   ```bash
   git clone https://github.com/pratikshyarout/AI-Chatbot.git
   cd AI-Chatbot
Create & activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate   # on Linux/macOS
venv\Scripts\activate      # on Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Configure environment variables:

Copy .env.example (if exists) or create .env

Add keys like OPENAI_API_KEY, VECTOR_DB_URL, etc.

Usage
To run the chatbot via Streamlit:

bash
Copy code
streamlit run streamlit_app.py
Access the UI in your browser (usually at http://localhost:8501).

Other modules (e.g. app.py) may also serve as entry points depending on your setup.

📈 How It Works (High Level)
User sends a message (text, or optionally media).

The message is embedded (embed.py) and used to query vector store (get_vector_db.py).

The system composes a prompt (via query.py), calls an LLM backend.

The response is returned, stored in memory (memory_store.py), and displayed in UI.

Optionally, the evaluation.py module assesses the response.
