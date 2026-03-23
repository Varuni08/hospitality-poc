from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from config import (
    GROQ_API_KEY, PINECONE_API_KEY,
    PINECONE_INDEX, EMBED_MODEL, GROQ_MODEL, PDF_PATH
)

groq_client = Groq(api_key=GROQ_API_KEY)
pc          = Pinecone(api_key=PINECONE_API_KEY)
embedder    = SentenceTransformer(EMBED_MODEL)
index       = pc.Index(PINECONE_INDEX)

GREETINGS = {"hi", "hello", "hey", "yo", "namaste", "vannakam", "howdy", "good morning", "good afteroon", "good evening", "good night", "gm"}

SYSTEM_PROMPT = """
You are the Information Agent for Saffron Table Bistro.
Answer questions about the restaurant using the context provided.
Topics: location, hours, parking, policies, events, general FAQs.
If the context doesn't contain the answer, say so politely.
Keep answers concise and friendly.
"""

def retrieve_chunks(query: str, top_k: int = 4) -> str:
    vec     = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    results = index.query(vector=vec, top_k=top_k, include_metadata=True)
    
    chunks = []
    for m in results.matches:
        meta = m["metadata"]
        text = meta.get("text", "")
        if not text:
            text = " | ".join(
                f"{k}: {v}" for k, v in meta.items() if v and v != "Nan"
            )
        if text:
            chunks.append(text)
    
    return "\n\n".join(chunks)

def run_info_agent(user_message: str, session: dict) -> str:
    if user_message.lower().strip() in GREETINGS:
        return "Hello! Welcome to Saffron Table Bistro. How can I help you today?"

    context = retrieve_chunks(user_message)

    history = session.get("info_history", [])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\nContext:\n{context}"}
    ] + history + [
        {"role": "user", "content": user_message}
    ]

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=messages
    )
    reply = resp.choices[0].message.content.strip()

    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": reply})
    session["info_history"] = history[-10:] 

    return reply
