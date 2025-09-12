# Mail_agent/Save_Reminder.py
from langchain_core.tools import tool
import json
from datetime import datetime

@tool
def save_reminder(task: str, time: str) -> str:
    """Save a reminder. The time must be in a valid future format 'YYYY-MM-DD HH:MM' processed by parse_natural_time."""
    # Assume time is pre-validated by parse_natural_time
    reminder = {
        "task": task,
        "time": time,
        "status": "pending",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    with open("reminders.json", "a") as f:
        f.write(json.dumps(reminder) + "\n")

    return f"✅ Reminder saved: '{task}' at {time}"