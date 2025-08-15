import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'local-rag')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

def get_vector_db(domain=None):
    """
    Get a Chroma vector database instance.
    If domain is specified, use a domain-specific collection.
    """
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    collection = f"{COLLECTION_NAME}-{domain}" if domain else COLLECTION_NAME
    db = Chroma(
        collection_name=collection,
        persist_directory=CHROMA_PATH,
        embedding_function=embedding,
    )
    return db
