"""
Configuration settings for the Enterprise Knowledge & Data Intelligence Assistant.
"""

from pathlib import Path
import os
import streamlit as st

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "documents"
DB_PATH = DATA_DIR / "enterprise.db"
MODELS_DIR = DATA_DIR / "models"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default Policy Path
DEFAULT_POLICY_PATH = DOCS_DIR / "company_policy.txt"

# LLM and Embedding Configuration
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# RAG & Similarity Hyperparameters
SIMILARITY_THRESHOLD = 0.38
SCORE_MARGIN = 0.15
TOP_K_CHUNKS = 4
MAX_CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# File Upload Limits
MAX_UPLOAD_SIZE_MB = 25
SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".csv", ".md"]

def get_gemini_api_key() -> str:
    """Retrieve the Gemini API key from Streamlit secrets or environment variables."""
    # 1. Try Streamlit secrets
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # 2. Try OS environment variable
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key

    # 3. Try checking local .env file
    try:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass

    return ""
