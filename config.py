import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

#đọc biến môi trường
load_dotenv()

#Qdrant config
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "")

#vector local dùng để embedding
VECTOR_SIZE: int = 1024

# ==========================================
# OPENROUTER CONFIG
# ==========================================
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_CHAT_MODEL: str = os.getenv("OPENROUTER_CHAT_MODEL")
OPENROUTER_EMBEDDING_MODEL: str = os.getenv("OPENROUTER_EMBEDDING_MODEL", "baai/bge-m3")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

_qdrant_client_instance = None

#hàm khởi tạo qdrant database
def get_qdrant_client() -> QdrantClient:
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        db_path = os.path.join(os.getcwd(), "local_qdrant_db")
        _qdrant_client_instance = QdrantClient(path=db_path)
    return _qdrant_client_instance

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
