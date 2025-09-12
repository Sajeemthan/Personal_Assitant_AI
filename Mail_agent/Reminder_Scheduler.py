# Mail_agent/Reminder_Scheduler.py
from langchain_core.tools import tool
import json
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

def _check_reminders() -> str:
    """Internal function to check for pending reminders (background use)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    updated_reminders = []
    output = []

    try:
        with open("reminders.json", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    for line in lines:
        reminder = json.loads(line)
        if reminder["status"] == "pending" and reminder["time"] == now:
            reminder["status"] = "done"
            task = reminder["task"]
            time = reminder["time"]
            output.append(f"🔔 Reminder: {task} (scheduled at {time})")
        updated_reminders.append(reminder)

    # Write back updated reminders
    with open("reminders.json", "w") as f:
        for reminder in updated_reminders:
            f.write(json.dumps(reminder) + "\n")

    return "\n".join(output) if output else "No reminders due at this time."

@tool
def check_reminders() -> str:
    """Tool for agent to manually check reminders."""
    return _check_reminders()

def start_scheduler():
    """Start the background scheduler to check reminders every minute."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(_check_reminders, "interval", minutes=1)  # Schedule the plain function
    scheduler.start()