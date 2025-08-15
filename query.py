import os
import time
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from get_vector_db import get_vector_db
from memory_store import load_memory, save_memory, get_relevant_memory
import tiktoken

load_dotenv()

HF_MODEL = os.getenv('HF_MODEL', 'sshleifer/tiny-gpt2')
RECENT_TURNS = int(os.getenv("RECENT_TURNS", "2"))
TOP_MEMORY_K = int(os.getenv("TOP_MEMORY_K", "3"))

_tokenizer = None
_model = None
_hf_pipe = None
_max_model_length = None

def _ensure_pipeline():
    global _tokenizer, _model, _hf_pipe, _max_model_length
    if _hf_pipe is None:
        model_name = HF_MODEL
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

        if any(x in model_name.lower() for x in ["t5", "flan", "bart"]):
            _model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            gen = pipeline("text2text-generation", model=_model, tokenizer=_tokenizer, max_new_tokens=256)
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            gen = pipeline("text-generation", model=_model, tokenizer=_tokenizer, max_new_tokens=256,
                           pad_token_id=_tokenizer.eos_token_id)

        _max_model_length = getattr(_model.config, "max_position_embeddings", 1024)
        _hf_pipe = HuggingFacePipeline(pipeline=gen)
        print(f"[INFO] Model loaded: {model_name}")
    return _hf_pipe

memory = load_memory()

def count_tokens(text):
    try:
        enc = tiktoken.encoding_for_model(HF_MODEL)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def get_prompts():
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="Generate 3 alternative phrasings of this question to improve search recall.\nReturn them as bullet points.\nOriginal: {question}"
    )
    template = (
        "You are a concise and helpful assistant for the {domain} domain.\n"
        "Use the Conversation History (for personalization & continuity) and the Retrieved Context (for factual grounding).\n"
        "If context is insufficient, say so briefly.\n\n"
        "Conversation History:\n{chat_history}\n\n"
        "Retrieved Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
    prompt = ChatPromptTemplate.from_template(template)
    return QUERY_PROMPT, prompt

def query(user_input, domain="general", return_docs=False):
    if not user_input:
        return "(no input)", 0 if return_docs else "(no input)"

    llm = _ensure_pipeline()
    db = get_vector_db(domain)
    QUERY_PROMPT, prompt = get_prompts()

    relevant_memories = get_relevant_memory(user_input, top_k=TOP_MEMORY_K)
    recent_history = memory.load_memory_variables({}).get("chat_history", [])[-(RECENT_TURNS * 2):]

    memory_snippets = [f"User: {m.get('user','')}\nBot: {m.get('bot','')}" for m in relevant_memories]
    for turn in recent_history:
        role = getattr(turn, "type", "User").capitalize()
        content = getattr(turn, "content", "")
        memory_snippets.append(f"{role}: {content}")
    combined_memory_text = "\n".join(memory_snippets)

    retriever = MultiQueryRetriever.from_llm(db.as_retriever(search_kwargs={"k": 2}), llm, prompt=QUERY_PROMPT)
    context_docs = retriever.get_relevant_documents(user_input)

    if not context_docs:
        context_docs = db.similarity_search(user_input, k=3)

    retrieved_context = "\n---\n".join([doc.page_content for doc in context_docs]) if context_docs else ""
    total_text = combined_memory_text + "\n" + retrieved_context + "\n" + user_input

    while count_tokens(total_text) > _max_model_length and memory_snippets:
        memory_snippets.pop(0)
        combined_memory_text = "\n".join(memory_snippets)
        total_text = combined_memory_text + "\n" + retrieved_context + "\n" + user_input

    while count_tokens(total_text) > _max_model_length and retrieved_context:
        words = retrieved_context.split()
        if len(words) <= 50:
            retrieved_context = ""
        else:
            retrieved_context = " ".join(words[50:])
        total_text = combined_memory_text + "\n" + retrieved_context + "\n" + user_input

    chain = (
        {
            "context": lambda _: retrieved_context,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: combined_memory_text,
            "domain": lambda _: domain,
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    response = str(chain.invoke(user_input))

    memory.save_context({"input": user_input}, {"output": response})
    save_memory(memory)

    if return_docs:
        return response, len(context_docs)
    return response
