import networkx as nx
from typing import Literal, Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai
import tempfile
import os
import mimetypes
import logging
import json
import re
import asyncio
import pytz
from datetime import datetime
from Calendar_agent.calendar_agent import process_user_input as calendar_process
from Mail_agent.mail_agent import process_mail_input as mail_process
from Mail_agent.Reminder_Scheduler import start_scheduler
from config import OPENAI_API_KEY
from history_manager import HistoryManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Start the reminder scheduler
start_scheduler()


# Initialize history manager
history_manager = HistoryManager(max_history=20)

# Initialize OpenAI client for audio transcription
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Define state for LangGraph
class AgentState(BaseModel):
    query: str
    user_id: str
    agent: Literal["orchestrator", "calendar", "mail", "casual", "none", "hybrid"] = "orchestrator"
    last_agent: Literal["calendar", "mail", "casual", "none", "hybrid"] = "none"
    response: str = ""
    email: str = ""
    metadata: Dict[str, Any] = {}
    event_details: Dict[str, Any] = {}
    next_agent: Literal["calendar", "mail", "casual", "none", "end"] = "end"  # For chaining

# Orchestrator prompt (LLM decides routing)
orchestrator_prompt = ChatPromptTemplate.from_template(
    """You are the main orchestrator for an AI assistant. Based on the user query and conversation history, decide which specialized agent should handle this request.

Available agents:
- calendar: For events, scheduling, dates, reminders, calendar-related tasks
- mail: For emails, inbox management, sending emails, Gmail-related tasks
- casual: For jokes, casual conversation, entertainment, general advice
- hybrid: For requests that require both calendar and mail actions (like creating an event and sending an email)
- none: If the request doesn't fit any category

Conversation History:
{history}

User Query: {query}

Output a JSON object with the following structure:
{{
  "agent": "calendar|mail|casual|hybrid|none",
  "reason": "Brief explanation of your decision",
  "next_agent": "calendar|mail|none|end"  // For hybrid flows, specify which agent to chain to next
}}"""
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)


async def extract_event_details_with_llm(query: str, history: str) -> Dict[str, Any]:
    """Extract event details (title, date, time) from a query using LLM."""
    prompt = ChatPromptTemplate.from_template(
        """Extract event details from the user query, considering conversation history.
        Today is {current_date}.
        
        Rules:
        1. Return a JSON object with: title (event name), date (YYYY-MM-DD), time (HH:MM if specified, else empty).
        2. Use history to infer missing details if present.
        3. If no event details found, return {"title": "", "date": "", "time": ""}.
        
        Examples:
        - Query: "Create meeting tomorrow at 3pm", History: "" → {"title": "meeting", "date": "2025-08-31", "time": "15:00"}
        - Query: "Email details for event", History: "User: Create meeting tomorrow" → {"title": "meeting", "date": "2025-08-31", "time": ""}
        - Query: "What's next?", History: "" → {"title": "", "date": "", "time": ""}
        
        Current Date: {current_date}
        Conversation History: {history}
        User Query: {query}
        
        Output ONLY the JSON object:"""
    )
    
    current_date = datetime.now(pytz.timezone('Asia/Colombo')).strftime("%Y-%m-%d %H:%M")
    try:
        response = await llm.ainvoke(prompt.format(
            current_date=current_date,
            history=history,
            query=query
        ))
        return json.loads(response.content.strip())
    except Exception as e:
        logger.error(f"Event detail extraction failed: {e}")
        return {"title": "", "date": "", "time": ""}

async def orchestrator_router(state: AgentState) -> AgentState:
    # Load conversation history
    history = history_manager.get_formatted_history(state.user_id, last_n=10)
    
    # Rule-based fallback for simple greetings
    query_lower = state.query.lower().strip()
    if query_lower in ["hello", "hi", "good morning", "hey"]:
        state.agent = "casual"
        state.response = f"{query_lower.capitalize()}, Sajee! I'm here to help with your emails or schedule. What would you like to do?"
        state.next_agent = "end"
        history_manager.add_message(state.user_id, "human", state.query)
        history_manager.add_message(state.user_id, "assistant", state.response)
        return state

    try:
        # Use LLM to decide routing
        response = await llm.ainvoke(orchestrator_prompt.format(
            history=history,
            query=state.query
        ))
        
        # Parse the JSON response
        decision = json.loads(response.content.strip())
        state.agent = decision.get("agent", "none")
        state.next_agent = decision.get("next_agent", "end")
        state.response = f"Routing to {state.agent} agent. Please wait..."  # Fallback response
    
    except Exception as e:
        logger.error(f"Orchestrator routing failed: {e}")
        state.agent = "none"
        state.response = "I encountered an error while processing your request. Please try again."
        state.next_agent = "end"
    
    # Add user query to history
    history_manager.add_message(state.user_id, "human", state.query)
    return state

async def generate_suggestion(history: List[Dict[str, Any]], last_action: str, current_response: str) -> str:
    """Generate context-aware suggestions using LLM"""
    prompt = ChatPromptTemplate.from_template(
    """Based on the conversation history, last action, and current response, suggest a natural follow-up question.
    If no suggestion is appropriate, output "NO_SUGGESTION".
    Examples:
    - History: "Create meeting tomorrow", Action: event_created, Response: "Meeting scheduled" → "Would you like to send an email about this meeting?"
    - History: "Send email to John", Action: email_about_meeting, Response: "Email sent" → "Would you like to add this to your calendar?"
    - History: "Tell me a joke", Action: none, Response: "Here's a joke" → "NO_SUGGESTION"
    Conversation History:
    {history}
    
    Last Action: {last_action}
    
    Current Response: {current_response}
    
    Suggestion:"""
    )
    
    # Format history
    history_str = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in history[-10:]])
    
    try:
        response = await llm.ainvoke(prompt.format(
            history=history_str,
            last_action=last_action,
            current_response=current_response
        ))
        suggestion = response.content.strip()
        return suggestion if suggestion != "NO_SUGGESTION" else ""
    except Exception as e:
        logger.error(f"Suggestion generation failed: {e}")
        return ""

# Then update the calendar_node and mail_node functions to use it:
async def calendar_node(state: AgentState) -> AgentState:
    """Process calendar requests"""
    try:
        # Get history for context
        history = history_manager.load_history(state.user_id)
        
        # Process with calendar agent
        response = await calendar_process(state.query, state.user_id, history)
        
        if isinstance(response, dict):
            state.response = response.get("reply", response.get("output", "Calendar action completed."))
            state.metadata = response.get("metadata", {})
            state.event_details = response.get("event_details", {})
        else:
            state.response = response
            
        # Generate suggestion
        suggestion = await generate_suggestion(history, state.metadata.get("action", ""), state.response)
        if suggestion:
            state.response += f"\n\n💡 Suggestion: {suggestion}"
            
    except Exception as e:
        logger.error(f"Calendar node failed: {e}")
        state.response = f"Sorry, I couldn't process your calendar request: {str(e)}"
    
    # Add response to history
    history_manager.add_message(state.user_id, "assistant", state.response, state.metadata)
    
    return state

async def mail_node(state: AgentState) -> AgentState:
    """Process mail requests"""
    try:
        # Get history for context
        history = history_manager.load_history(state.user_id)
        
        # Process with mail agent
        response = await mail_process(state.query, state.user_id, history)
        
        if isinstance(response, dict):
            state.response = response.get("output", response.get("reply", "Mail action completed."))
            state.metadata = response.get("metadata", {})
        else:
            state.response = response
            
        # Generate suggestion
        suggestion = await generate_suggestion(history, state.metadata.get("action", ""), state.response)
        if suggestion:
            state.response += f"\n\n💡 Suggestion: {suggestion}"
            
    except Exception as e:
        logger.error(f"Mail node failed: {e}")
        state.response = f"Sorry, I couldn't process your mail request: {str(e)}"
    
    # Add response to history
    history_manager.add_message(state.user_id, "assistant", state.response, state.metadata)
    
    return state

async def casual_node(state: AgentState) -> AgentState:
    """Process casual requests"""
    try:
        # Get history for context
        history = history_manager.load_history(state.user_id)
        
        # For casual requests, we'll use the mail agent which has casual handling
        response = await mail_process(state.query, state.user_id, history)
        
        if isinstance(response, dict):
            state.response = response.get("output", response.get("reply", "I'm here to help!"))
        else:
            state.response = response
            
    except Exception as e:
        logger.error(f"Casual node failed: {e}")
        state.response = f"Sorry, I couldn't process your request: {str(e)}"
    
    # Add response to history
    history_manager.add_message(state.user_id, "assistant", state.response)
    
    return state

# Add conditional edges
def decide_next_node(state: AgentState) -> str:
    """Decide which node to go to next based on the agent type"""
    if state.agent == "calendar":
        return "calendar"
    elif state.agent == "mail":
        return "mail"
    elif state.agent == "casual":
        return "casual"
    elif state.agent == "hybrid":
        # For hybrid flows, we need to handle chaining
        if state.next_agent == "calendar":
            return "calendar"
        elif state.next_agent == "mail":
            return "mail"
        else:
            return "end"
    else:
        return "end"

# Set up the graph
workflow = StateGraph(AgentState)
workflow.add_node("orchestrator", orchestrator_router)
workflow.add_node("calendar", calendar_node)
workflow.add_node("mail", mail_node)
workflow.add_node("casual", casual_node)
workflow.set_entry_point("orchestrator")
workflow.add_conditional_edges(
    "orchestrator",
    decide_next_node
)
workflow.add_edge("calendar", END)
workflow.add_edge("mail", END)
workflow.add_edge("casual", END)
app.graph = workflow.compile()


# Define input model
class ProcessRequest(BaseModel):
    query: str
    user_id: str = "default_user"

@app.post("/process")
async def process_request(request: ProcessRequest) -> Dict[str, str]:
    logger.info(f"Received process request for user {request.user_id}: {request.query}")
    try:
        # Load history
        history = history_manager.get_formatted_history(request.user_id, last_n=10)
        
        # Initialize state
        state = AgentState(query=request.query, user_id=request.user_id, agent="orchestrator")
        
        # Invoke the graph
        if not hasattr(app, 'graph') or app.graph is None:
            raise HTTPException(status_code=500, detail="Graph workflow not initialized")
        
        result = await app.graph.ainvoke(state.model_dump())  
        
        # Ensure result is a dict and has response
        if isinstance(result, dict):
            response = result.get('response', "Sorry, I couldn't process your request.")
            metadata = result.get('metadata', {})
        else:
            raise HTTPException(status_code=500, detail="Workflow did not return a valid response")
        
        # Add to history
        history_manager.add_message(request.user_id, "human", request.query)
        history_manager.add_message(request.user_id, "assistant", response, metadata)
        
        return {"response": response}
    
    except HTTPException as e:
        logger.error(f"HTTP Error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return {"response": f"Error: {str(e)}. Please try again."}


# FastAPI endpoint for audio transcription
@app.post("/transcribe-audio")
async def transcribe_audio(audio: UploadFile = File(...), user_id: str = "default_user") -> Dict[str, str]:
    logger.info(f"Received audio transcription request for user {user_id}")
    try:
        valid_audio_types = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/mp4', 'audio/x-m4a']
        file_ext = os.path.splitext(audio.filename)[1].lower() if audio.filename else ''
        valid_extensions = ['.mp3', '.wav', '.m4a']
        content_type = audio.content_type

        if content_type == 'application/octet-stream' and file_ext in valid_extensions:
            content_type = mimetypes.guess_type(audio.filename)[0] or 'audio/mpeg'
        elif content_type not in valid_audio_types or file_ext not in valid_extensions:
            raise HTTPException(status_code=400, detail="Please upload a valid audio file (e.g., .mp3, .wav, .m4a).")

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext or '.m4a') as temp_file:
            content = await audio.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty audio file")
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            with open(temp_file_path, "rb") as audio_file:
                transcription = openai_client.audio.transcriptions.create(model="whisper-1", file=audio_file, language="en")
                transcribed_text = transcription.text.strip()
        finally:
            os.unlink(temp_file_path)

        if not transcribed_text:
            return {"transcription": "", "response": "I couldn't understand the audio. Could you speak clearly or try typing instead?"}

        # Process the transcribed text
        state = AgentState(
            query=transcribed_text, 
            user_id=user_id,
            agent="orchestrator"
        )
        
        result = await app.graph.ainvoke(state)
        
        # Add to history
        history_manager.add_message(user_id, "human", transcribed_text)
        if result.response:
            history_manager.add_message(user_id, "assistant", result.response, result.metadata)
            
        return {"transcription": transcribed_text, "response": result.response or "Sorry, I couldn't process your request."}

    except HTTPException:
        raise
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "insufficient_quota" in error_str:
            return {"transcription": "", "response": "Error: OpenAI API quota exceeded. Please add a payment method."}
        logger.error(f"Error processing audio: {error_str}")
        return {"transcription": "", "response": f"Error processing your audio: {error_str}. Please try again."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)