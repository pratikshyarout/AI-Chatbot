# seed_data.py
import os
from embed import embed

# Folder containing your local PDFs
PDF_FOLDER = os.path.join(os.path.dirname(__file__), "seed_docs")  # e.g., backend/pdfs

def seed_local_pdfs():
    if not os.path.exists(PDF_FOLDER):
        print(f"[!] PDF folder not found: {PDF_FOLDER}")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("[!] No PDF files found in seed folder.")
        return

    for pdf in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf)
        print(f"[+] Embedding {pdf} ...")
        embed(pdf_path, domain="general")

if __name__ == "__main__":
    seed_local_pdfs()
