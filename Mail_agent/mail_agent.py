# Mail_agent/mail_agent.py
import os
import json
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from datetime import datetime, timedelta 
from langchain_core.tools import tool
from config import OPENAI_API_KEY, GOOGLE_CREDENTIALS_PATH
from Mail_agent.Save_Reminder import save_reminder
from Mail_agent.Reminder_Scheduler import check_reminders
from Mail_agent.GetTime_Tool import GetCurrentTimeTool
import re

# Gmail credentials setup with folder-specific token
credentials = get_gmail_credentials(
    token_file="token_mail.json",  # Folder-specific token
    scopes=["https://mail.google.com/"],
    client_secrets_file=GOOGLE_CREDENTIALS_PATH,  # Shared root credentials
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

# Update instructions to use history context
instructions = """
You are an AI assistant helping Sajee manage his Gmail inbox, email communications, and calendar tasks effectively.
You have access to tools that can read, send, and organize emails, set reminders, and manage calendar events through an integrated system.

Key updates:
- Use the conversation history to maintain context and avoid asking for repeated information
- For hybrid queries from calendar, check history for event details
- After sending emails about meetings, naturally suggest calendar events based on context
- Use history to understand the flow of conversation and provide appropriate suggestions

🔍 Inbox Monitoring and Prioritization:
1. Monitor unread emails using the Gmail API.
2. Classify each email as **Important** or **Unimportant** based on:
   - Sender domain (e.g., "@client.com", "@gov.lk").
   - Keywords (e.g., "urgent", "approval", "quotation", "proposal", "payment due", "action required").
   - Tone and sentiment of the message (e.g., demanding, serious, formal).
   - Presence of attachments, especially quotations, proposals, or invoices.
3. Mark emails containing **quotations** or **proposals** as **Important**, regardless of tone or sender.
4. Label and sort the emails accordingly for better filtering.

📣 Summaries:
5. Create a summary of emails in a concise format, for example:  
   - "Emails summary: from whom, subject, summary of email"
   if the user asks for important emails summary, only give important email's summary.
- For recap commands (e.g., "summarize unread emails"), use search_gmail with query='is:unread' or 'label:inbox', max_results as needed (e.g., 5 for last 5), classify importance, and list in bullets.
- For queries like "recap of the last 5 emails" or "summary of the last 5 emails", use search_gmail with query='label:inbox', max_results=5, then summarize each email with sender, subject, date, and a short snippet/body summary.
- For queries like "summary of emails from [sender]", use search_gmail with query='from:[sender]', then summarize the results.
- Always use search_gmail for any email recap or summary, and generate a concise, bullet-point summary from the results.

✍️ Sending Emails:
7. When the user requests to send or draft an email:
   - Do NOT use tools like `get_gmail_message`, `search_gmail`, or any fetch tools unless explicitly needed (e.g., to reference an existing email).
   - ALWAYS prompt for and collect ALL required fields BEFORE drafting or sending: recipient (must be a full email address like name@domain.com—ask if only name is given), subject, and message body. If any field is missing or incomplete (e.g., no domain in email), ask polite follow-up questions to fill in the gaps.
   - Use `get_current_time` to resolve relative dates/times (e.g., 'tomorrow') only if it works synchronously; if it fails or causes an error, skip it, assume today is the current system date (e.g., 2025-08-14), calculate 'tomorrow' accordingly (e.g., 2025-08-15), and notify the user.
   - Step-by-step for drafting/sending: 1. Collect/confirm ALL fields (ask if needed). 2. Resolve dates with `get_current_time` or fallback to current system date, then compose and SHOW the full draft text. 3. Ask for review/changes. 4. ONLY after user confirmation, use tools to save as draft or send. If any step fails, show the draft with the fallback date and pause.
8. Compose the email with a professional and concise tone. Add the signature:
   **"Best regards,\nSajee."**
   ALWAYS show the full email draft to the user for review BEFORE saving as draft or sending, and wait for confirmation. If the user says "send", use the appropriate tool only after showing and confirming.
🛑 Sending Rules and Review:
11. Do not send emails automatically.
12. Always wait for user reviews and confirmations before sending anything.
13. If the user chooses to proceed with sending a drafted email, ask:
   **"Would you like to send the draft email now, make any changes, or leave it as is?"**

✅ Tone and Style:
14. Be concise, clear, and professional in all messages.
15. Prioritize clarity, organization, and respectful interaction.

🎯 Overall Goal:
Your main goal is to help Sajee manage emails and calendar tasks intelligently, never miss a high-priority message or event, and maintain professional communication—all while keeping him in control. 
For greetings or casual chats (e.g., 'Hi', 'Good morning'), respond warmly and mention both email and calendar assistance, e.g., 'Good morning, Sajee! I'm here to help with your emails or schedule. What would you like to do?' 
For casual queries like requesting a joke or relaxation (e.g., 'tell me a joke', 'help me relax', 'I got depression, tell me a joke'), immediately respond with a lighthearted joke and a positive message, e.g., 'Sure, Sajee! Why did the tomato turn red? Because it saw the salad dressing! I hope that helps—let me know how else I can assist.' Do not use tools for these queries; provide a direct response instead. Avoid limiting responses to email-only topics unless the query is email-specific.
- For pinning or starring emails (treat 'star' as 'pin'), if the email is described by subject, sender, or content (e.g., 'pin email about payment from John'), first use search_gmail to find the matching email ID, then use pin_email with that ID. For showing pinned, use show_pinned_emails directly.
When user provides details like names/emails (e.g., 'John's email is john123@gmail.com'), use save_contact tool to store. For sending, use get_contact_email to recall. Parse full input for multiple details (e.g., reminder + email in one message).
Carefully parse the full user input for multiple details at once (e.g., if input has reminder time AND email, extract both). Use regex or direct extraction for emails (e.g., match [\w\.-]+@[\w\.-]+) and names. Confirm all fields before drafting.

NEW: For emails about meetings or scheduling, always suggest creating a calendar event as well.
"""

# Pull and customize prompt
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

# Initialize LLM with rate limit awareness
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY, max_retries=2)

# Define tools
tools = toolkit.get_tools()
tools.extend([save_reminder, check_reminders, GetCurrentTimeTool()])

# NEW: Define pinning tools with correct @tool decorator
@tool
def pin_email(email_id: str) -> str:
    """Pin an email by ID."""
    try:
        with open("pinned_items.json", "r+") as f:
            data = json.load(f)
            if "mail" not in data:
                data["mail"] = []
            if email_id not in data["mail"]:  # Avoid duplicates
                data["mail"].append(email_id)
                f.seek(0)
                json.dump(data, f)
                f.truncate()
        return "Email pinned successfully."
    except FileNotFoundError:
        with open("pinned_items.json", "w") as f:
            json.dump({"mail": [email_id]}, f)
        return "Email pinned (new file created)."
    except json.JSONDecodeError:
        with open("pinned_items.json", "w") as f:
            json.dump({"mail": [email_id]}, f)
        return "Email pinned (file corrupted, recreated)."
    except Exception as e:
        return f"Failed to pin email: {str(e)}. Please try again later."

@tool
def show_pinned_emails() -> str:
    """Show pinned emails."""
    try:
        with open("pinned_items.json", "r") as f:
            data = json.load(f)
            pinned_emails = data.get("mail", [])
            if not pinned_emails:
                return "No pinned emails."
            return "\n".join(pinned_emails)
    except FileNotFoundError:
        return "No pinned emails yet."
    except json.JSONDecodeError:
        return "Error reading pinned emails (file corrupted)."
    except Exception as e:
        return f"Failed to show pinned emails: {str(e)}. Please try again later."

# Extend tools list with new pinning tools
tools.extend([pin_email, show_pinned_emails])

# NEW: Define contact management tools
@tool
def save_contact(name: str, email: str) -> str:
    """Save a contact's email by name."""
    try:
        with open("contacts.json", "r+") as f:
            data = json.load(f)
            data[name.lower()] = email
            f.seek(0)
            json.dump(data, f)
        return f"Saved {name}'s email as {email}."
    except FileNotFoundError:
        with open("contacts.json", "w") as f:
            json.dump({name.lower(): email}, f)
        return f"Created and saved {name}'s email as {email}."

@tool
def get_contact_email(name: str) -> str:
    """Get a contact's email by name."""
    try:
        with open("contacts.json", "r") as f:
            data = json.load(f)
            return data.get(name.lower(), "Email not found.")
    except FileNotFoundError:
        return "No contacts saved yet."

# Extend tools list with new contact tools
tools.extend([save_contact, get_contact_email])

# Create agent
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# Update process_mail_input to accept history
async def process_mail_input(user_input: str, user_id: str = "default_user", history: list = None) -> dict:
    """Process user input for mail agent with history context."""
    # Format history for context
    history_str = ""
    if history:
        history_str = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                               for msg in history[-20:]])  # Last messages
    
    # Check if this is a follow-up from calendar
    is_calendar_followup = False
    calendar_event_details = {}
    
    if history:
        # Look for recent calendar events in history
        for msg in reversed(history[-3:]):  # Check last 3 messages
            if msg.get('role') == 'assistant' and msg.get('metadata', {}).get('action') == 'event_created':
                is_calendar_followup = True
                calendar_event_details = msg.get('metadata', {})
                break
    
    try:
        # Add history context to the input
        enhanced_input = user_input
        if is_calendar_followup and "email" in user_input.lower():
            enhanced_input = f"{user_input} Event details: {json.dumps(calendar_event_details)}"
        
        # Include history in the input for the agent
        full_input = f"Conversation history:\n{history_str}\n\nCurrent request: {enhanced_input}"
        
        print(f"Full input to agent: {full_input}")

        response = await agent_executor.ainvoke({"input": full_input})
        print(f"Agent response: {response}")
        output = response["output"]
        metadata = {}
        
        # Extract metadata for suggestions
        output_lower = output.lower()
        if "reminder saved" in output_lower:
            metadata["action"] = "reminder_saved"
            task_match = re.search(r"Reminder: '(.+?)' at", output) or re.search(r"saved: '(.+?)' at", output)
            if task_match:
                metadata["task"] = task_match.group(1).strip()
        elif "summary" in user_input.lower() and ("meeting" in output_lower or "appointment" in output_lower):
            metadata["action"] = "email_summary_with_meeting"
        elif "meeting" in output_lower or "appointment" in output_lower or "schedule" in output_lower:
            metadata["action"] = "email_about_meeting"
            # Extract recipient name for suggestion
            name_match = re.search(r'to\s+(\w+)', output_lower) or re.search(r'for\s+(\w+)', output_lower)
            if name_match:
                metadata["recipient"] = name_match.group(1).strip()
        
        return {"output": output, "metadata": metadata}
    except Exception as e:
        error_str = str(e)
        metadata = {}
        if "429" in error_str or "insufficient_quota" in error_str:
            return {"output": "Error: OpenAI API quota exceeded. Please add a payment method.", "metadata": metadata}
        return {"output": f"Sorry, something went wrong: {str(e)}. Please try again or rephrase.", "metadata": metadata}
    
async def run_mail_agent(user_input: str, user_id: str = "default_user") -> str:
    """Alias for process_mail_input to support test scripts."""
    result = await process_mail_input(user_input, user_id)
    return result["output"]