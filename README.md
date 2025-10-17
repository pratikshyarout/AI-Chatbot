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

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+  
- API keys / credentials (for embedding or language model services)  
- A vector database or embedding provider (FAISS, Pinecone, etc.)


📈 How It Works (High Level)
User sends a message (text, or optionally media).

The message is embedded (embed.py) and used to query vector store (get_vector_db.py).

The system composes a prompt (via query.py), calls an LLM backend.

The response is returned, stored in memory (memory_store.py), and displayed in UI.

Optionally, the evaluation.py module assesses the response.
