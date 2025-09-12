# history_manager.py
import json
import os
from typing import List, Dict, Any
from datetime import datetime

class HistoryManager:
    def __init__(self, max_history=20):
        self.max_history = max_history
        self.history_file = "history.json"
        
    def load_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Load history for a specific user"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    all_history = json.load(f)
                    return all_history.get(user_id, [])
            return []
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def save_history(self, user_id: str, history: List[Dict[str, Any]]):
        """Save history for a specific user, maintaining only max_history items"""
        # Ensure we only keep the last max_history items
        if len(history) > self.max_history:
            history = history[-self.max_history:]
            
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    all_history = json.load(f)
            else:
                all_history = {}
                
            all_history[user_id] = history
            
            with open(self.history_file, 'w') as f:
                json.dump(all_history, f, indent=2)
                
        except (json.JSONDecodeError, FileNotFoundError):
            with open(self.history_file, 'w') as f:
                json.dump({user_id: history}, f, indent=2)
    
    def add_message(self, user_id: str, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to history"""
        history = self.load_history(user_id)
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        self.save_history(user_id, history)
        return history
    
    def get_formatted_history(self, user_id: str, last_n: int = 10) -> str:
        """Get formatted history string for LLM context"""
        history = self.load_history(user_id)
        if not history:
            return "No history yet."
            
        # Get last n messages
        recent_history = history[-last_n:] if len(history) > last_n else history
        
        formatted = []
        for msg in recent_history:
            prefix = "User" if msg["role"] == "human" else "Assistant"
            formatted.append(f"{prefix}: {msg['content']}")
            
        return "\n".join(formatted)