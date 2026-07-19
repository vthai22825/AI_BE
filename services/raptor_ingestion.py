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

def _summarize_cluster(items_in_cluster: list[dict]) -> str:
    # Gộp văn bản kèm theo thông tin trang
    combined_text = ""
    for item in items_in_cluster:
        combined_text += f"\n--- Trang {item['page_number']} ---\n{item['text']}"

    prompt = f"""Bạn là một chuyên gia tổng hợp thông tin. Nhiệm vụ của bạn là đọc các đoạn trích dưới đây và viết một bản tóm tắt toàn diện nhưng súc tích.
    
YÊU CẦU QUAN TRỌNG:
1. Giữ lại tất cả các từ khóa, định nghĩa, thực thể (tên riêng, địa điểm, số liệu) và các khái niệm cốt lõi.
2. Trình bày rõ ràng mối liên hệ giữa các ý chính để nội dung có thể được dùng làm tài liệu tra cứu (RAG).
3. Không thêm thắt thông tin bên ngoài, chỉ dựa vào văn bản được cung cấp.
4. TRÍCH DẪN TẠI CHỖ (INLINE CITATIONS): Khi bạn tóm tắt một ý hoặc một đoạn, BẮT BUỘC phải chèn một liên kết markdown trích dẫn nguồn trang ngay tại dòng đó (ở cuối câu hoặc đoạn). 
Cú pháp trích dẫn BẮT BUỘC: `[Trang X](#page-X)` hoặc `[Trang X, Trang Y](#page-X)`
Ví dụ: "Hệ thống hỗ trợ tâm lý rất quan trọng đối với Thẩm phán [Trang 10](#page-10)."

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


def _generate_global_summary(parent_texts: list[str], parent_cited_pages: list[list[int]]) -> str:
    """Tóm tắt lại toàn bộ các parent_texts thành một Global Summary"""
    combined_text = ""
    for text, pages in zip(parent_texts, parent_cited_pages):
        # Tạo chuỗi "Trang 1, 2" hoặc "N/A"
        pages_str = ", ".join(map(str, pages)) if pages else "N/A"
        combined_text += f"\n--- Nguồn từ các trang: {pages_str} ---\n{text}"
    
    prompt = f"""Bạn là một chuyên gia tổng hợp tài liệu cấp cao.
Dưới đây là các bản tóm tắt của từng phần trong một tài liệu (có kèm số trang). Nhiệm vụ của bạn là gộp chúng lại và viết một BẢN TÓM TẮT TOÀN CỤC bao quát toàn bộ tài liệu.

YÊU CẦU NGHIÊM NGẶT:
1. TUYỆT ĐỐI TRUNG THÀNH VỚI VĂN BẢN: Không được tự ý sáng tạo, không chèn thêm bất kỳ kiến thức cá nhân nào.
2. BẢO TOÀN THUẬT NGỮ: Không được khái quát hóa các danh sách quan trọng.
3. TRÍCH DẪN TẠI CHỖ (INLINE CITATIONS): Khi bạn tóm tắt một ý hoặc một đoạn, BẮT BUỘC phải chèn một liên kết markdown trích dẫn nguồn trang ngay tại dòng đó (ở cuối câu hoặc đoạn). 
Cú pháp trích dẫn BẮT BUỘC: `[Trang X](#page-X)` hoặc `[Trang X, Trang Y](#page-X)`
Ví dụ: "Hệ thống hỗ trợ tâm lý rất quan trọng đối với Thẩm phán [Trang 10](#page-10)."
4. Trình bày cấu trúc rõ ràng (Sử dụng tiêu đề, danh sách gạch đầu dòng, bảng biểu nếu cần).

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
def raptor_ingest(pages: list[dict], document_id: str, session_id: str, qdrant_client: QdrantClient)-> dict[str, Any]:

    from langchain_core.documents import Document
    docs = []
    for page in pages:
        docs.append(Document(page_content=page["text"], metadata={"page_number": page["page_number"]}))

    #chunkings text thành các đoạn
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )
    chunk_docs = text_splitter.split_documents(docs)

    if not chunk_docs:
        raise ValueError("Lỗi khi phân tách đoạn văn bản, không tạo được chunks")
    
    leaf_ids = [str(uuid.uuid4()) for _ in chunk_docs]
    chunks = [doc.page_content for doc in chunk_docs]
    page_numbers = [doc.metadata.get("page_number", "N/A") for doc in chunk_docs]
    
    #khởi tạo collection qdrant
    ensure_collection_exists(qdrant_client)
    from qdrant_client.http.models import PointStruct
    
    #embedding chunks
    leaf_vectors = _embed_texts(chunks)

    #clustering và tạo nút cha
    labels = _cluter_vertor(leaf_vectors)

    cluster_map: dict[int, list[dict]] = {}
    for idx, label in enumerate(labels):
        cluster_map.setdefault(int(label), []).append({
            "text": chunks[idx],
            "leaf_id": leaf_ids[idx],
            "page_number": page_numbers[idx]
        })
    
    #summarize nút cha
    parent_texts = []
    parent_leaf_ids = []
    parent_cited_pages = []
    
    for cluster_id, items in sorted(cluster_map.items()):
        summary = _summarize_cluster(items)
        parent_texts.append(summary)
        
        parent_leaf_ids.append([item["leaf_id"] for item in items])
        # Lọc unique page_numbers (loại bỏ N/A)
        pages = list(set([item["page_number"] for item in items if item["page_number"] != "N/A"]))
        parent_cited_pages.append(pages)
    
    #embedding nút cha
    parent_vectors = _embed_texts(parent_texts)

    #tạo global summary (level 2)
    global_summary = _generate_global_summary(parent_texts, parent_cited_pages)
    global_vector = _embed_texts([global_summary])[0]

    #lưu vào trong qdrant
    points = []
    for chunk, leaf_id, vector, page_num in zip(chunks, leaf_ids, leaf_vectors, page_numbers):
        points.append(PointStruct(
            id=leaf_id, 
            vector=vector, 
            payload={
                "session_id": session_id, 
                "document_id": document_id, 
                "level": 0, 
                "text_content": chunk,
                "page_number": page_num
            }
        ))
        
    for p_text, p_vector, p_leaf_ids, p_pages in zip(parent_texts, parent_vectors, parent_leaf_ids, parent_cited_pages):
        points.append(PointStruct(
            id=str(uuid.uuid4()), 
            vector=p_vector, 
            payload={
                "session_id": session_id, 
                "document_id": document_id, 
                "level": 1, 
                "text_content": p_text,
                "leaf_ids": p_leaf_ids,
                "cited_pages": p_pages
            }
        ))

    # lưu global summary
    all_pages = list(set([p for p in page_numbers if p != "N/A"]))
    points.append(PointStruct(
        id=str(uuid.uuid4()), 
        vector=global_vector, 
        payload={
            "session_id": session_id, 
            "document_id": document_id, 
            "level": 2, 
            "text_content": global_summary,
            "type": "global_summary",
            "cited_pages": all_pages
        }
    ))

    qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)

    logger.info("Lưu vào database thành công")
    return {"total_leaf_nodes": len(chunks), "total_parent_nodes": len(parent_texts), "has_global_summary": True}