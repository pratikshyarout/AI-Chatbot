import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="TurboAsk", page_icon="⚡", layout="wide")

# --- Chat style CSS ---
st.markdown("""
<style>
.chat-container { display:flex; flex-direction:column; gap:12px; }
.user-msg { align-self:flex-end; background:#DCF8C6; color:#000; padding:10px 16px; border-radius:20px 20px 0 20px; max-width:70%; word-wrap:break-word; }
.assistant-msg { align-self:flex-start; background:#2E2E2E; color:#FFF; padding:10px 16px; border-radius:20px 20px 20px 0; max-width:70%; word-wrap:break-word; }
.latency-text { font-size:10px; color:#AAA; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#FFDD00;'>⚡ TurboAsk</h1>"
            "<p style='text-align:center; color:#888;'>Multimodal Conversational RAG (Text • Image • Audio • Video)</p>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    domain = st.selectbox("Choose Domain", ["law","healthcare","finance","education","multimodal","general"], index=5)

    st.markdown("---")
    if st.button("🧹 Clear All Memory (File + Session)"):
        try:
            r = requests.post(f"{API_URL}/clear_memory")
            st.success(r.json().get("message", "Memory cleared"))
        except Exception as e:
            st.error(f"Failed to clear memory: {e}")

    if st.button("🔄 Forget Current Session"):
        try:
            r = requests.post(f"{API_URL}/forget_session")
            st.success(r.json().get("message", "Session forgotten"))
        except Exception as e:
            st.error(f"Failed to forget session: {e}")

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

def render_message(role, content, latency=None, retrieved_docs=None):
    bubble_class = "user-msg" if role=="user" else "assistant-msg"
    latency_text = f"<div class='latency-text'>⏱ {latency:.2f}s | 📄 Retrieved Docs: {retrieved_docs}</div>" if latency else ""
    st.markdown(f"<div class='chat-container'><div class='{bubble_class}'>{content}{latency_text}</div></div>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("latency"), msg.get("retrieved_docs"))

# --- User input ---
user_input = st.chat_input("💬 Ask me anything...")

st.markdown("**Attach optional media (image/audio/video):**")
uploads = st.file_uploader("", type=[
    "png","jpg","jpeg","bmp","tiff",
    "mp3","wav","m4a","flac","ogg",
    "mp4","mov","avi","mkv","webm"
], accept_multiple_files=True)

# --- Send query ---
if user_input or uploads:
    if user_input:
        st.session_state.messages.append({"role":"user","content":user_input})
        render_message("user", user_input)

    with st.spinner("🤖 TurboAsk is thinking..."):
        files_payload = [("files", (f.name, f.getvalue())) for f in uploads] if uploads else None
        data = {"query": user_input or "", "domain": domain}
        try:
            if uploads:
                resp = requests.post(f"{API_URL}/query", data=data, files=files_payload, timeout=600)
            else:
                resp = requests.post(f"{API_URL}/query", json=data, timeout=600)

            if resp.ok:
                res_json = resp.json()
                answer = res_json.get("answer", "(no answer)")
                latency = res_json.get("latency", None)
                retrieved_docs = res_json.get("retrieved_docs", None)
            else:
                answer = f"Error {resp.status_code}: {resp.text}"
                latency = None
                retrieved_docs = None
        except Exception as e:
            answer = f"Request failed: {e}"
            latency = None
            retrieved_docs = None

    st.session_state.messages.append({"role":"assistant","content":answer,"latency":latency,"retrieved_docs":retrieved_docs})
    render_message("assistant", answer, latency, retrieved_docs)
