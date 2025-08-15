# embed.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from get_vector_db import get_vector_db

# Chunk size safe for flan-t5-base (~400 characters)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

def embed(file_path, domain="general"):
    """
    Embeds the given PDF file into the vector database.
    file_path: path to the PDF file
    domain: vector DB domain name (default: general)
    """
    try:
        if not os.path.exists(file_path):
            print(f"[!] File not found: {file_path}")
            return False

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split into small chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            print(f"[!] No text extracted from {file_path}")
            return False

        db = get_vector_db(domain)
        db.add_documents(chunks)
        print(f"[+] Embedded {len(chunks)} chunks from {os.path.basename(file_path)}")
        return True

    except Exception as e:
        print(f"[!] Failed to embed {os.path.basename(file_path)}: {e}")
        return False
