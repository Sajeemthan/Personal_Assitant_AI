# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env")

GOOGLE_CREDENTIALS_PATH = "credentials.json"  # Assuming fixed path at root; can be os.getenv if dynamic

