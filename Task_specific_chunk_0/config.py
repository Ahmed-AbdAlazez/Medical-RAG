from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
CHROMA_DIR = BASE_DIR / "chroma_db"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 3
HYBRID_SEMANTIC_WEIGHT = 0.65
HYBRID_KEYWORD_WEIGHT = 0.35
JINA_MODEL_NAME = "jina-embeddings-v2-base-en"
TFIDF_MAX_FEATURES = 512
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")