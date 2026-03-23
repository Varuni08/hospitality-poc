import re
import json
from datetime import datetime
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from supabase import create_client
from config import (
    GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX,
    EMBED_MODEL, GROQ_MODEL, SUPABASE_URL, SUPABASE_KEY
)

groq_client = Groq(api_key=GROQ_API_KEY)
pc          = Pinecone(api_key=PINECONE_API_KEY)
embedder    = SentenceTransformer(EMBED_MODEL)
supabase    = create_client(SUPABASE_URL, SUPABASE_KEY)
index       = pc.Index(PINECONE_INDEX)

SYSTEM_PROMPT = """
You are the Reservation Agent for Saffron Table Bistro.

CRITICAL RULES:
- NEVER save a reservation unless you have ALL of these: name, phone, date, time, guests
- Collect them ONE BY ONE in this exact order: name → phone → date → time → guests → special requests
- If the user skips a field, ask for it again before moving on
- Short replies like a name, number, or date are ALWAYS answers to your last question
- Only generate the save_reservation JSON block when ALL 5 required fields are collected

You help guests:
- Make new reservations
- Look up existing reservations (by name, phone, or reservation ID)
- Modify reservations
- Cancel reservations
- Request callbacks

When ready to save, respond with this JSON block:
{"action": "save_reservation", "data": {"name": "", "phone": "", "date": "", "time": "", "guests": "", "special_requests": ""}}

For other actions:
{"action": "find_reservation", "query": "<name or phone or ID>"}
{"action": "modify_reservation", "id": "<RES_ID>", "field": "<Field>", "value": "<val>"}
{"action": "cancel_reservation", "id": "<RES_ID>"}
{"action": "save_callback", "data": {"name": "", "phone": "", "reason": ""}}
{"action": "none"}
"""

def save_reservation(data: dict) -> str:
    res_id = f"RES{datetime.now().strftime('%y%m%d%H%M%S')}"
    supabase.table("reservations").insert({
        "id":               res_id,
        "name":             data.get("name", ""),
        "phone":            data.get("phone", ""),
        "date":             data.get("date", ""),
        "time":             data.get("time", ""),
        "guests":           str(data.get("guests", "")),
        "special_requests": data.get("special_requests", ""),
        "status":           "Confirmed"
    }).execute()
    vec = embedder.encode(
        f"{data.get('name')} {data.get('date')} {data.get('time')}"
    ).tolist()
    index.upsert(vectors=[{
        "id":       res_id,
        "values":   vec,
        "metadata": {"name": data.get("name"), "date": data.get("date"), "id": res_id}
    }])
    return res_id


def find_reservation(query: str):
    for col, val in [("id", query.upper()), ("phone", query)]:
        res = supabase.table("reservations").select("*").eq(col, val).execute()
        if res.data:
            return res.data[0]
    res = supabase.table("reservations").select("*").ilike("name", f"%{query}%").execute()
    return res.data[0] if res.data else None


def modify_reservation(res_id: str, field: str, value: str) -> bool:
    col_map = {
        "Date": "date", "Time": "time",
        "Guests": "guests", "Special Requests": "special_requests"
    }
    col = col_map.get(field)
    if col:
        supabase.table("reservations").update({col: value}).eq("id", res_id).execute()
        return True
    return False


def cancel_reservation(res_id: str):
    supabase.table("reservations").update({"status": "Cancelled"}).eq("id", res_id).execute()


def save_callback(data: dict) -> str:
    ref = f"CB{datetime.now().strftime('%y%m%d%H%M%S')}"
    supabase.table("callbacks").insert({
        "ref":    ref,
        "name":   data.get("name", ""),
        "phone":  data.get("phone", ""),
        "reason": data.get("reason", "")
    }).execute()
    return ref


def execute_action(action_block: dict) -> str:
    action = action_block.get("action", "none")

    if action == "save_reservation":
        res_id = save_reservation(action_block["data"])
        return f"Reservation confirmed! Your ID is **{res_id}**."

    elif action == "find_reservation":
        rec = find_reservation(action_block["query"])
        if rec:
            return (
                f"Found reservation **{rec['id']}** for {rec['name']} — "
                f"{rec['date']} at {rec['time']} for {rec['guests']} guests. "
                f"Status: {rec['status']}."
            )
        return "No reservation found with that information."

    elif action == "modify_reservation":
        ok = modify_reservation(action_block["id"], action_block["field"], action_block["value"])
        return "Reservation updated!" if ok else "Could not update — invalid field."

    elif action == "cancel_reservation":
        cancel_reservation(action_block["id"])
        return f"Reservation {action_block['id']} has been cancelled."

    elif action == "save_callback":
        ref = save_callback(action_block["data"])
        return f"Callback request logged! Reference: **{ref}**. We'll call you soon."

    return ""


def run_reservation_agent(user_message: str, session: dict) -> str:
    if "reservation_history" not in session:
        session["reservation_history"] = []

    history = session["reservation_history"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ] + history + [
        {"role": "user", "content": user_message}
    ]

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        messages=messages
    )
    reply = resp.choices[0].message.content.strip()

    json_match = re.search(r'\{.*?"action".*?\}', reply, re.DOTALL)
    if json_match:
        try:
            action_block  = json.loads(json_match.group())
            action_result = execute_action(action_block)
            if action_result:
                reply = re.sub(r'\{.*?"action".*?\}', action_result, reply, flags=re.DOTALL)
        except json.JSONDecodeError:
            pass

    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": reply})
    session["reservation_history"] = history[-10:]

    return reply
