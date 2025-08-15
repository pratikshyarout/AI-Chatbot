import os
import time
import shutil
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from embed import embed
import query as query_module
from memory_store import save_memory, create_memory, clear_memory_file
from media_processing import extract_text_from_file
from get_vector_db import get_vector_db  # ✅ for DB check

load_dotenv()

TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
RESET_DB = os.getenv('RESET_DB_ON_STARTUP', 'false').lower() == 'true'

os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Optional DB reset
if RESET_DB:
    shutil.rmtree('./chroma', ignore_errors=True)
    print("[INFO] Vector DB reset on startup.")

# ✅ Check and seed Chroma DB before loading the model
try:
    db = get_vector_db("general")
    if not db.get()["ids"]:  # DB empty
        print("[*] Chroma DB empty — running seed_data.py from local PDFs...")
        os.system("python seed_data.py")  # Will read local PDFs, no internet needed
    else:
        print(f"[*] Chroma DB already has {len(db.get()['ids'])} documents.")
except Exception as e:
    print(f"[!] Could not check/seed DB: {e}")

app = Flask(__name__)
CORS(app)

# ===== Preload the model at startup =====
print("🔄 Loading model into memory... please wait...")
query_module._ensure_pipeline()
print("✅ Model loaded and ready to accept queries!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": query_module._hf_pipe is not None}), 200

@app.route('/embed', methods=['POST'])
def route_embed():
    domain = request.form.get('domain', 'general')
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if embed(file, domain=domain):
        return jsonify({"message": f"File embedded successfully into '{domain}' domain"}), 200
    return jsonify({"error": "File embedding failed"}), 400

@app.route('/query', methods=['POST'])
def route_query():
    start_time = time.time()
    query_text = None
    domain = 'general'
    extracted_context = []

    if request.content_type and 'application/json' in request.content_type:
        data = request.get_json(silent=True) or {}
        query_text = data.get('query', '')
        domain = data.get('domain', 'general')
    else:
        query_text = request.form.get('query', '')
        domain = request.form.get('domain', 'general')
        files = request.files.getlist('files') or ([] if 'file' not in request.files else [request.files['file']])
        for f in files:
            if not f or not f.filename:
                continue
            path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
            f.save(path)
            text = extract_text_from_file(path)
            if text:
                extracted_context.append(text)
            try:
                os.remove(path)
            except Exception:
                pass

    if not query_text and not extracted_context:
        return jsonify({"error": "No query or extractable media provided"}), 400

    merged = query_text
    if extracted_context:
        merged += "\n\n[MEDIA CONTEXT]\n" + "\n---\n".join(extracted_context)

    response, retrieved_docs = query_module.query(merged, domain=domain, return_docs=True)

    latency = time.time() - start_time
    print(f"[API METRICS] Query: {query_text[:50]}... | Domain: {domain} | Latency: {latency:.2f}s")

    if response:
        return jsonify({
            "answer": response,
            "latency": latency,
            "retrieved_docs": retrieved_docs
        }), 200
    return jsonify({"error": "Something went wrong"}), 400

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    query_module.memory = create_memory()
    save_memory(query_module.memory)
    clear_memory_file()
    return jsonify({"message": "Memory cleared"}), 200

@app.route('/forget_session', methods=['POST'])
def forget_session():
    query_module.memory = create_memory()
    return jsonify({"message": "Current session forgotten"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
