import json
from groq import Groq
from agents.info_agent import run_info_agent
from agents.menu_agent import run_menu_agent
from agents.reservation_agent import run_reservation_agent
from config import GROQ_API_KEY, GROQ_MODEL


class MultiAgentOrchestrator:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

        self.agent_registry = {
            "info_agent":        run_info_agent,
            "menu_agent":        run_menu_agent,
            "reservation_agent": run_reservation_agent,
        }

        self.router_prompt = """
You are a routing agent for a restaurant AI assistant.
Your only job is to decide which internal agent should handle the user's query.

Available agents:
1. info_agent
   - greetings, timings, location, parking
   - delivery, smoking area, restaurant policies, general FAQ

2. menu_agent
   - cuisines, appetizers, mains, desserts
   - veg / non-veg / vegan, dishes, ratings
   - best pairings, alcohol / drinks

3. reservation_agent
   - reserve a table, modify reservation
   - cancel reservation, callback request
   - booking questions

Rules:
- Return only valid JSON, no extra text.
- If one agent is enough, return one.
- If query combines multiple tasks, return multiple.
- Greetings go to info_agent.
- Booking always includes reservation_agent.
- Menu exploration always includes menu_agent.

Return strictly in this format:
{"agents": ["info_agent"], "reason": "short reason"}
"""

    def keyword_route(self, user_message: str):
        msg = user_message.lower()

        reservation_keywords = [
            "book", "booking", "reserve", "reservation", "table for",
            "cancel reservation", "modify reservation", "change reservation",
            "callback", "call me back"
        ]
        menu_keywords = [
            "menu", "dish", "dishes", "cuisine", "cuisines", "food",
            "appetizer", "starter", "main course", "main", "dessert",
            "veg", "vegan", "non veg", "non-veg", "alcohol", "wine",
            "beer", "cocktail", "drink", "pairing"
        ]
        info_keywords = [
            "location", "address", "where are you", "timing", "timings",
            "hours", "open", "close", "parking", "delivery", "smoking",
            "policy", "contact", "phone"
        ]

        selected = []
        if any(k in msg for k in reservation_keywords):
            selected.append("reservation_agent")
        if any(k in msg for k in menu_keywords):
            selected.append("menu_agent")
        if any(k in msg for k in info_keywords):
            selected.append("info_agent")

        if selected:
            return {
                "agents": list(dict.fromkeys(selected)),
                "reason": "keyword routing"
            }
        return None

    def llm_route(self, user_message: str):
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": self.router_prompt},
                {"role": "user",   "content": user_message}
            ]
        )
        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
            if "agents" in parsed and isinstance(parsed["agents"], list):
                return parsed
        except Exception:
            pass
        return {"agents": ["info_agent"], "reason": "fallback routing"}

    def route(self, user_message: str):
        keyword_result = self.keyword_route(user_message)
        if keyword_result:
            return keyword_result
        return self.llm_route(user_message)

    def combine_outputs(self, outputs: dict) -> str:
        valid = [v for v in outputs.values() if v]
        return "\n\n".join(valid) if valid else "Sorry, I couldn't process that."

    def handle(self, user_message: str, session: dict) -> str:
        word_count = len(user_message.strip().split())
        is_short_reply = word_count <= 3

        if "active_agent" in session and is_short_reply:
            selected_agents = [session["active_agent"]]
        elif "active_agent" in session:
            active  = session["active_agent"]
            routing = self.route(user_message)
            new_agent = routing.get("agents", [active])[0]
            if new_agent != active:
                session.pop("active_agent", None)
                selected_agents = routing.get("agents", ["info_agent"])
            else:
                selected_agents = [active]
        else:
            routing = self.route(user_message)
            selected_agents = routing.get("agents", ["info_agent"])

        if len(selected_agents) == 1:
            session["active_agent"] = selected_agents[0]

        print(f"\n Router → {selected_agents}")

        outputs = {}
        for agent_name in selected_agents:
            agent_fn = self.agent_registry.get(agent_name)
            if agent_fn:
                outputs[agent_name] = agent_fn(user_message, session)

        return self.combine_outputs(outputs)
