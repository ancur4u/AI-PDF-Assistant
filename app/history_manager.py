import json
import os

HISTORY_FILE = "data/chat_history.json"

def save_to_history(query, response):
    os.makedirs("data", exist_ok=True)
    history = load_history()
    history.append({"query": query, "response": response})
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []