import os
import json
from langchain.memory import ConversationBufferMemory

# Path to store memory
MEMORY_PATH = "chat_memory.json"

def save_memory(memory):
    """
    Save conversation memory to a JSON file.
    """
    try:
        messages = []
        for m in memory.chat_memory.messages:
            messages.append({
                "type": m.__class__.__name__,
                "content": m.content
            })

        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)

        print(f"[INFO] Memory saved to {MEMORY_PATH}")

    except Exception as e:
        print(f"[ERROR] Failed to save memory: {e}")


def load_memory():
    """
    Load conversation memory from the JSON file, if it exists.
    """
    try:
        if os.path.exists(MEMORY_PATH):
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                messages_data = json.load(f)

            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            for m in messages_data:
                if m["type"] == "HumanMessage":
                    from langchain.schema import HumanMessage
                    memory.chat_memory.add_user_message(m["content"])
                elif m["type"] == "AIMessage":
                    from langchain.schema import AIMessage
                    memory.chat_memory.add_ai_message(m["content"])
            print(f"[INFO] Memory loaded from {MEMORY_PATH}")
            return memory
    except Exception as e:
        print(f"[ERROR] Failed to load memory: {e}")

    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def create_memory():
    """
    Create a new ConversationBufferMemory.
    """
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)
def get_relevant_memory(query, top_k=3):
    """
    Retrieve the most relevant memory entries for the given query.
    This is a placeholder that just returns the last `top_k` messages.
    You can replace this with an embedding-based similarity search.
    """
    try:
        if os.path.exists("memory_store.json"):
            with open("memory_store.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            return data[-top_k:] if data else []
    except Exception as e:
        print(f"[ERROR] Failed to get relevant memory: {e}")
    return []



def clear_memory_file():
    """
    Deletes the persistent memory file from disk.
    """
    try:
        if os.path.exists(MEMORY_PATH):
            os.remove(MEMORY_PATH)
            print(f"[INFO] Memory file '{MEMORY_PATH}' cleared.")
        else:
            print("[INFO] No memory file found to clear.")
    except Exception as e:
        print(f"[ERROR] Failed to clear memory file: {e}")
