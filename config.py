import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

#đọc biến môi trường
load_dotenv()

#Qdrant config
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "")

#vector local dùng để embedding
EMBEDDING_MODEL : str = os.getenv("EMBEDDING_MODEL","")
VECTOR_SIZE: int = 1536

# ==========================================
# 1. GITHUB MODELS CONFIG
# ==========================================
GITHUB_API_KEY: str = os.getenv("GITHUB_API_KEY", "")
GITHUB_BASE_URL: str = "https://models.inference.ai.azure.com"

GITHUB_FALLBACK_MODELS: list[str] = [
    "gpt-4o",
    "claude-3-5-sonnet",
    "Meta-Llama-3.1-70B-Instruct",
    "gpt-4o-mini"
]

# ==========================================
# 2. GOOGLE GEMINI CONFIG
# ==========================================
GOOGLE_API_KEYS: list[str] = [
    os.getenv("GOOGLE_API_KEY_1", ""),
    os.getenv("GOOGLE_API_KEY_2", ""),
    os.getenv("GOOGLE_API_KEY_3", "")
]

GOOGLE_API_KEYS = [key for key in GOOGLE_API_KEYS if key]

FALLBACK_MODELS: list[str] = [
    "gemini-2.5-pro",
    "gemini-2.5-flash"
]


OLLAMA_SUMMARY_MODEL: str = os.getenv("OLLAMA_SUMMARY_MODEL","")

#hàm khởi tạo qdrant database
def get_qdrant_client() -> QdrantClient:
    #khởi tạo qdrant db
    db_path = os.path.join(os.getcwd(), "local_qdrant_db")

    #khởi tạo client 
    client = QdrantClient(path=db_path)
    return client

#hàm khởi tạo collection (nếu chưa có)
def ensure_collection_exists(client: QdrantClient) -> None:
    if not client.collection_exists(QDRANT_COLLECTION_NAME):
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION_NAME,
            field_name="workspace_id",
            field_schema="keyword"
        )

#raptor config
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "300"))
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
