from __future__ import annotations
import logging
import time
import uuid
from typing import Any
import numpy as np
from qdrant_client import QdrantClient
from config import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    ensure_collection_exists, 
    OLLAMA_SUMMARY_MODEL, 
    GOOGLE_API_KEYS, 
    QDRANT_COLLECTION_NAME,
    GITHUB_API_KEY,
    GITHUB_BASE_URL,
    GITHUB_FALLBACK_MODELS
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


logger = logging.getLogger(__name__)

#embedding bằng mô hình nhúng của google
if not GOOGLE_API_KEYS:
    raise ValueError("Không tìm thấy GOOGLE_API_KEY nào trong cấu hình.")
google_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2", 
    google_api_key=GOOGLE_API_KEYS[0],
    output_dimensionality=1536
)

def _embed_texts(chunks: list[str], batch_size: int =100)-> list[list[float]]:
    #vì google limit bản free nên nhúng theo batch
    all_vector = []
    total_chunks = len(chunks)
    if total_chunks == 0:
        return []
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]

        try:
            batch_vector = google_embeddings.embed_documents(batch_chunks)
            all_vector.extend(batch_vector)
            
            if i + batch_size < total_chunks:
                time.sleep(1)
        except Exception as e:
            logger.error("Lỗi khi nhúng tại lô %d: %s", (i // batch_size) + 1, str(e))
            raise e
    return all_vector


#hàm co cụm
def _cluter_vertor(vectors: list[list[float]])-> np.ndarray:
    #số lượng cụm là tổng vector chia 6 lấy phần nguyên (cân bằng tốc độ và chi tiết)
    n_samples = len(vectors)
    effective_clusters = max(1, n_samples // 6)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=effective_clusters, random_state=42, n_init="auto")
    return kmeans.fit_predict(np.array(vectors))

#hàm summarise
def _summarize_cluster(texts_in_cluster: list[str]) -> str:
    # Gộp văn bản (để nguyên độ dài vì đã chia nhỏ từ trước)
    combined_text = "\n---\n".join(texts_in_cluster)

    prompt = f"""Bạn là một chuyên gia tổng hợp thông tin. Nhiệm vụ của bạn là đọc các đoạn trích dưới đây và viết một bản tóm tắt toàn diện nhưng súc tích.
    
YÊU CẦU QUAN TRỌNG:
1. Giữ lại tất cả các từ khóa, định nghĩa, thực thể (tên riêng, địa điểm, số liệu) và các khái niệm cốt lõi.
2. Trình bày rõ ràng mối liên hệ giữa các ý chính để nội dung có thể được dùng làm tài liệu tra cứu (RAG).
3. Không thêm thắt thông tin bên ngoài, chỉ dựa vào văn bản được cung cấp.

Văn bản cần tóm tắt:
{combined_text}"""
    
    try:
        from langchain_ollama import ChatOllama
        
        logger.info(f"Đang tóm tắt bằng OLLAMA model: {OLLAMA_SUMMARY_MODEL}")
        
        llm = ChatOllama(
            model=OLLAMA_SUMMARY_MODEL, 
            temperature=0.3,      
            num_predict=2048,      
            timeout=120.0,         
        )
            
        response = llm.invoke(prompt)
        return response.content.strip()
    
    except Exception as e:
        logger.error(f"Ollama ({OLLAMA_SUMMARY_MODEL}) gặp lỗi khi tóm tắt: {e}")
        return "Lỗi: Không thể tóm tắt đoạn văn bản này bằng AI nội bộ."


def _generate_global_summary(parent_texts: list[str]) -> str:
    """Tóm tắt lại toàn bộ các parent_texts thành một Global Summary"""
    combined_text = "\n---\n".join(parent_texts)
    
    prompt = f"""Bạn là một chuyên gia tổng hợp tài liệu cấp cao.
Dưới đây là các bản tóm tắt của từng phần trong một tài liệu. Nhiệm vụ của bạn là gộp chúng lại và viết một BẢN TÓM TẮT TOÀN CỤC bao quát toàn bộ tài liệu.

YÊU CẦU NGHIÊM NGẶT:
1. TUYỆT ĐỐI TRUNG THÀNH VỚI VĂN BẢN: Không được tự ý sáng tạo, không chèn thêm bất kỳ tên công cụ (như eslint, prettier...), công nghệ, hay kiến thức cá nhân nào không xuất hiện trong nội dung được cung cấp.
2. BẢO TOÀN THUẬT NGỮ: Không được khái quát hóa các danh sách quan trọng. (Ví dụ: Nếu văn bản gốc liệt kê cụ thể XSS, SQL Injection, IDOR... thì phải giữ nguyên tên của chúng, không được gộp chung thành "các lỗi bảo mật").
3. Trình bày cấu trúc rõ ràng (Sử dụng tiêu đề, danh sách gạch đầu dòng, bảng biểu nếu cần).

Nội dung các phần:
{combined_text}"""
    
    try:
        from langchain_openai import ChatOpenAI
        
        # Thử sử dụng các Cloud Models (GPT-4o, Claude...)
        if GITHUB_API_KEY:
            for model_name in GITHUB_FALLBACK_MODELS:
                try:
                    logger.info("Đang sinh Global Summary bằng Cloud AI: %s...", model_name)
                    llm = ChatOpenAI(
                        model=model_name,
                        api_key=GITHUB_API_KEY,
                        base_url=GITHUB_BASE_URL,
                        temperature=0.0, # Hạ nhiệt độ xuống 0.0 để tránh ảo giác
                        max_retries=0
                    )
                    response = llm.invoke(prompt)
                    return response.content.strip()
                except Exception as e:
                    logger.warning("Truy xuất qua %s thất bại: %s. Kích hoạt model dự phòng...", model_name, e)
                    import asyncio
                    # dùng time.sleep thay vì asyncio.sleep vì hàm này là đồng bộ (sync)
                    time.sleep(1)
            
            logger.error("Đã thử toàn bộ Cloud Models nhưng đều thất bại. Quay về dùng Local AI.")
        
        # Nếu không có GITHUB_API_KEY hoặc tất cả cloud fail thì fallback về Ollama
        from langchain_ollama import ChatOllama
        logger.info(f"Đang sinh Global Summary bằng Local AI ({OLLAMA_SUMMARY_MODEL})...")
        llm = ChatOllama(
            model=OLLAMA_SUMMARY_MODEL, 
            temperature=0.3,      
            num_predict=4096,      
            timeout=180.0,         
        )
        response = llm.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"Lỗi khi sinh Global Summary: {e}")
        return "Lỗi: Không thể tạo bản tóm tắt toàn cục."

#hàm thực thi raptor
def raptor_ingest(text: str, document_id: str, session_id: str, qdrant_client: QdrantClient)-> dict[str, Any]:

    #chunkings text thành các đoạn
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(text)

    if not chunks:
        raise ValueError("Lỗi khi phân tách đoạn văn bản, không tạo được chunks")
    
    #khởi tạo collection qdrant
    ensure_collection_exists(qdrant_client)
    from qdrant_client.http.models import PointStruct
    
    #embedding chunks
    leaf_vectors = _embed_texts(chunks)

    #clustering và tạo nút cha
    labels = _cluter_vertor(leaf_vectors)

    cluster_map: dict[int, list[str]] = {}
    for idx, label in enumerate(labels):
        cluster_map.setdefault(int(label), []).append(chunks[idx])
    
    #summarize nút cha
    parent_texts = []
    
    for cluster_id, texts in sorted(cluster_map.items()):
        summary = _summarize_cluster(texts)
        parent_texts.append(summary)
        # đã xóa time.sleep(2) để tăng tốc độ
    
    #embedding nút cha
    parent_vectors = _embed_texts(parent_texts)

    #tạo global summary (level 2)
    global_summary = _generate_global_summary(parent_texts)
    global_vector = _embed_texts([global_summary])[0]

    #lưu vào trong qdrant
    points = []
    for t, vector in zip(chunks, leaf_vectors):
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"session_id": session_id, "document_id": document_id, "level": 0, "text_content": t}))
        
    for t, vector in zip(parent_texts, parent_vectors):
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"session_id": session_id, "document_id": document_id, "level": 1, "text_content": t}))

    # lưu global summary
    points.append(PointStruct(
        id=str(uuid.uuid4()), 
        vector=global_vector, 
        payload={
            "session_id": session_id, 
            "document_id": document_id, 
            "level": 2, 
            "text_content": global_summary,
            "type": "global_summary"
        }
    ))

    qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)

    logger.info("Lưu vào database thành công")
    return {"total_leaf_nodes": len(chunks), "total_parent_nodes": len(parent_texts), "has_global_summary": True}