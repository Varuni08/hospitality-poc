import pandas as pd
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from config import (
    GROQ_API_KEY, PINECONE_API_KEY,
    PINECONE_INDEX, EMBED_MODEL, GROQ_MODEL, MENU_CSV_PATH
)

groq_client = Groq(api_key=GROQ_API_KEY)
pc          = Pinecone(api_key=PINECONE_API_KEY)
embedder    = SentenceTransformer(EMBED_MODEL)
index       = pc.Index(PINECONE_INDEX)

ALCOHOL_DISCLAIMER = (
    "Alcoholic beverages are served only to guests above 21. "
    "A valid ID may be required on the premises."
)

SYSTEM_PROMPT = """
You are the Menu Agent for Saffron Table Bistro.
Help guests explore the menu: dishes, prices, ingredients,
dietary options (veg, vegan, gluten-free) and drinks.
Answer using only the menu context provided.
Be friendly and specific.
"""

def retrieve_menu_chunks(query: str, top_k: int = 6) -> str:
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
    
def run_menu_agent(user_message: str, session: dict) -> str:
    if "menu_state" not in session:
        session["menu_state"] = {
            "record_type":  None,
            "cuisine":      None,
            "category":     None,
            "diet":         None,
            "beverage_type": None
        }

    msg = user_message.lower()

    is_alcohol = any(word in msg for word in ["alcohol", "wine", "beer", "cocktail", "drink"])

    context = retrieve_menu_chunks(user_message)

    if is_alcohol:
        context += f"\n\n{ALCOHOL_DISCLAIMER}"

    history = session.get("menu_history", [])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\nMenu context:\n{context}"}
    ] + history + [
        {"role": "user", "content": user_message}
    ]

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=messages
    )
    reply = resp.choices[0].message.content.strip()

    if is_alcohol:
        reply += f"\n\n {ALCOHOL_DISCLAIMER}"

    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": reply})
    session["menu_history"] = history[-10:]

    return reply
