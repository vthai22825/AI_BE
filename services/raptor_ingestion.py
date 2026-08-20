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
    OPENROUTER_API_KEY, 
    OPENROUTER_CHAT_MODEL,
    OPENROUTER_EMBEDDING_MODEL,
    OPENROUTER_BASE_URL,
    QDRANT_COLLECTION_NAME
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

logger = logging.getLogger(__name__)

# embedding bằng mô hình nhúng của OpenRouter
if not OPENROUTER_API_KEY:
    raise ValueError("Không tìm thấy OPENROUTER_API_KEY trong cấu hình.")

openrouter_embeddings = OpenAIEmbeddings(
    model=OPENROUTER_EMBEDDING_MODEL, 
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    chunk_size=1 # Bắt buộc là 1 vì BGE-M3 trên OpenRouter chỉ nhận string, không nhận array
)

def _embed_texts(chunks: list[str], batch_size: int = 100) -> tuple[list[list[float]], int]:
    # Thư viện Langchain luôn bọc chữ trong mảng ["text"] kể cả khi dùng embed_query
    # Do đó ta dùng thư viện requests thuần tuý để đảm bảo gửi đúng định dạng string cho OpenRouter bge-m3
    all_vector = []
    total_chunks = len(chunks)
    if total_chunks == 0:
        return [], 0
        
    logger.info(f"Đang tiến hành nhúng {total_chunks} đoạn văn bản (Direct HTTP Request)...")
    try:
        from concurrent.futures import ThreadPoolExecutor
        import requests
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        def get_embedding(chunk: str):
            clean_chunk = chunk.strip()
            if not clean_chunk:
                clean_chunk = "[EMPTY_CONTENT]"
                
            payload = {
                "model": OPENROUTER_EMBEDDING_MODEL,
                "input": clean_chunk # Gửi ĐÚNG 1 string thuần, không bọc mảng
            }
            res = requests.post(f"{OPENROUTER_BASE_URL}/embeddings", headers=headers, json=payload)
            if res.status_code != 200:
                raise ValueError(f"OpenRouter API Error: {res.text}")
            
            data = res.json()
            usage = data.get("usage", {}).get("total_tokens", 0)
            return data["data"][0]["embedding"], usage
            
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(get_embedding, chunks))
            all_vector = [r[0] for r in results]
            total_tokens = sum(r[1] for r in results)
            
    except Exception as e:
        logger.error(f"Lỗi khi nhúng văn bản qua HTTP: {str(e)}")
        raise e

    # Chuẩn hóa tất cả các vector về cùng độ dài 1024 (của BGE-M3) để tránh lỗi inhomogeneous shape
    normalized_vectors = []
    for v in all_vector:
        if not isinstance(v, list):
            normalized_vectors.append([0.0] * 1024)
        elif len(v) < 1024:
            normalized_vectors.append(v + [0.0] * (1024 - len(v)))
        elif len(v) > 1024:
            normalized_vectors.append(v[:1024])
        else:
            normalized_vectors.append(v)
            
    return normalized_vectors, total_tokens


def _cluter_vertor(vectors: list[list[float]]) -> np.ndarray:
    # Đảm bảo tất cả các vector có cùng kích thước trước khi đưa vào KMeans
    if not vectors:
        return np.array([])
        
    # Tìm độ dài phổ biến nhất (thường là 1024)
    try:
        lengths = [len(v) if isinstance(v, list) else 0 for v in vectors]
        from collections import Counter
        most_common_len = Counter([l for l in lengths if l > 0]).most_common(1)[0][0]
        
        valid_vectors = []
        for v in vectors:
            if not isinstance(v, list):
                valid_vectors.append([0.0] * most_common_len)
            elif len(v) < most_common_len:
                # Pad with zeros
                valid_vectors.append(v + [0.0] * (most_common_len - len(v)))
            elif len(v) > most_common_len:
                # Truncate
                valid_vectors.append(v[:most_common_len])
            else:
                valid_vectors.append(v)
                
        vectors = valid_vectors
    except Exception as e:
        logger.error(f"Lỗi khi chuẩn hóa vector: {e}")

    # số lượng cụm là tổng vector chia 6 lấy phần nguyên (cân bằng tốc độ và chi tiết)
    n_samples = len(vectors)
    effective_clusters = max(1, n_samples // 6)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=effective_clusters, random_state=42, n_init="auto")
    return kmeans.fit_predict(np.array(vectors))

def _summarize_cluster(items_in_cluster: list[dict]) -> tuple[str, int]:
    # Gộp văn bản kèm theo thông tin trang
    combined_text = ""
    for item in items_in_cluster:
        combined_text += f"\n--- Trang {item['page_number']} ---\n{item['text']}"

    prompt = f"""Bạn là một chuyên gia tổng hợp thông tin học thuật. Nhiệm vụ của bạn là đọc các đoạn trích dưới đây và viết một bản tóm tắt toàn diện, súc tích, đi thẳng vào vấn đề cốt lõi.
    
YÊU CẦU QUAN TRỌNG:
1. LOẠI BỎ THÔNG TIN NHIỄU (ANTI-SPAM): Tuyệt đối KHÔNG đưa vào bản tóm tắt các thông tin rác lặp lại ở đầu/cuối mỗi trang (ví dụ: tên tác giả, tên trung tâm, số điện thoại, bản quyền, địa chỉ web). Chỉ tập trung vào CHUYÊN MÔN và KIẾN THỨC.
2. BẢNG BIỂU CHO DỮ LIỆU CẤU TRÚC: BẤT CỨ KHI NÀO có các cặp số liệu, tập hợp mẫu thống kê, danh sách tham số... BẮT BUỘC trình bày dưới dạng Bảng Markdown `| Cột 1 | Cột 2 |`. KHÔNG liệt kê số liệu liên tiếp bằng dấu phẩy trong ngoặc đơn.
3. ĐỊNH DẠNG TOÁN HỌC (LaTeX): Mọi biến số toán học (X, Y), công thức, ký hiệu xác suất, phương trình BẮT BUỘC phải bọc trong dấu `$` (inline) hoặc `$$` (block) để hiển thị chuẩn xác.
4. TRÍCH DẪN TẠI CHỖ (INLINE CITATIONS): Khi tóm tắt một ý, BẮT BUỘC chèn liên kết markdown trích dẫn nguồn trang ngay tại dòng đó. Nếu trình bày Bảng, hãy đặt trích dẫn ngay trên tiêu đề bảng.
- Cú pháp trích dẫn: `[Trang X](#page-X)` hoặc `[Trang X, Trang Y](#page-X)`.
- Không lặp lại trích dẫn liên tục nếu cùng một trang.

Văn bản cần tóm tắt:
{combined_text}"""
    
    try:
        llm = ChatOpenAI(
            model=OPENROUTER_CHAT_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.2,      
            max_retries=1
        )
            
        response = llm.invoke(prompt)
        usage = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
        return response.content.strip(), usage
    
    except Exception as e:
        logger.error(f"OpenRouter ({OPENROUTER_CHAT_MODEL}) gặp lỗi khi tóm tắt: {e}")
        return "Lỗi: Không thể tóm tắt đoạn văn bản này bằng AI.", 0


def _generate_global_summary(parent_texts: list[str], parent_cited_pages: list[list[int]]) -> tuple[str, int]:
    """Tóm tắt lại toàn bộ các parent_texts thành một Global Summary"""
    combined_text = ""
    for text, pages in zip(parent_texts, parent_cited_pages):
        # Tạo chuỗi "Trang 1, 2" hoặc "N/A"
        pages_str = ", ".join(map(str, pages)) if pages else "N/A"
        combined_text += f"\n--- Nguồn từ các trang: {pages_str} ---\n{text}"
    
    prompt = f"""Bạn là một chuyên gia tổng hợp tài liệu cấp cao. Dưới đây là các bản tóm tắt của từng phần trong một tài liệu. Nhiệm vụ của bạn là gộp chúng lại và viết một BẢN TÓM TẮT TOÀN CỤC bao quát toàn bộ tài liệu.

YÊU CẦU NGHIÊM NGẶT VỀ ĐỊNH DẠNG VÀ NỘI DUNG:
1. LOẠI BỎ THÔNG TIN NHIỄU: Tuyệt đối KHÔNG liệt kê tên tác giả, số điện thoại, tên trung tâm hoặc các thông tin bản quyền ở header/footer. Đi thẳng vào kiến thức chuyên môn.
2. GỘP TRÍCH DẪN THÔNG MINH (ANTI-SPAM CITATIONS): KHÔNG BAO GIỜ tạo ra một chuỗi trích dẫn khổng lồ kiểu `[Trang 1, Trang 2... Trang 50]`. Nếu một thông tin quá chung chung xuất hiện ở mọi trang, HÃY BỎ QUA trích dẫn cho thông tin đó. Cú pháp trích dẫn: `[Trang X](#page-X)`.
3. SỬ DỤNG BẢNG (MARKDOWN TABLES): Khi gặp các danh sách số liệu, so sánh tham số, dữ liệu tương quan, BẮT BUỘC dùng thẻ Bảng `| Cột 1 | Cột 2 |` để trình bày rõ ràng.
4. TOÁN HỌC VÀ THỐNG KÊ (LaTeX): Mọi công thức, ký hiệu toán học, biến số phải được bọc trong `$` hoặc `$$`.
5. CẤU TRÚC UI THOÁNG ĐÃNG: 
- Bắt buộc tạo các Tiêu đề con (`###`, `####`) phân cấp rõ ràng.
- Sử dụng Quote `>` cho các định nghĩa quan trọng.
- Bắt buộc có 1 dòng trống (blank line) giữa các đoạn văn, tiêu đề và bảng biểu.

Nội dung các phần:
{combined_text}"""
    
    try:
        logger.info(f"Đang sinh Global Summary bằng OpenRouter Model: {OPENROUTER_CHAT_MODEL}...")
        llm = ChatOpenAI(
            model=OPENROUTER_CHAT_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.0, # Hạ nhiệt độ xuống 0.0 để tránh ảo giác
            max_retries=1
        )
        response = llm.invoke(prompt)
        usage = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
        return response.content.strip(), usage
        
    except Exception as e:
        logger.error(f"Lỗi khi sinh Global Summary: {e}")
        return "Lỗi: Không thể tạo bản tóm tắt toàn cục.", 0

# hàm thực thi raptor
def raptor_ingest(pages: list[dict], document_id: str, session_id: str, qdrant_client: QdrantClient) -> dict[str, Any]:

    from langchain_core.documents import Document
    docs = []
    for page in pages:
        docs.append(Document(page_content=page["text"], metadata={"page_number": page["page_number"]}))

    # chunkings text thành các đoạn
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunk_docs = text_splitter.split_documents(docs)

    if not chunk_docs:
        raise ValueError("Lỗi khi phân tách đoạn văn bản, không tạo được chunks")
    
    leaf_ids = [str(uuid.uuid4()) for _ in chunk_docs]
    chunks = [doc.page_content for doc in chunk_docs]
    page_numbers = [doc.metadata.get("page_number", "N/A") for doc in chunk_docs]
    
    # khởi tạo collection qdrant
    ensure_collection_exists(qdrant_client)
    from qdrant_client.http.models import PointStruct
    
    total_tokens_used = 0
    # embedding chunks
    leaf_vectors, leaf_tokens = _embed_texts(chunks)
    total_tokens_used += leaf_tokens

    # clustering và tạo nút cha
    labels = _cluter_vertor(leaf_vectors)

    cluster_map: dict[int, list[dict]] = {}
    for idx, label in enumerate(labels):
        cluster_map.setdefault(int(label), []).append({
            "text": chunks[idx],
            "leaf_id": leaf_ids[idx],
            "page_number": page_numbers[idx]
        })
    
    # summarize nút cha bằng Đa luồng (Multi-threading)
    parent_texts = []
    parent_leaf_ids = []
    parent_cited_pages = []
    
    logger.info(f"Đang tóm tắt {len(cluster_map)} cụm bằng OpenRouter ({OPENROUTER_CHAT_MODEL}) đa luồng...")
    
    from concurrent.futures import ThreadPoolExecutor
    
    # Chạy song song các hàm tóm tắt
    def process_cluster(cluster_id, items):
        summary, tokens = _summarize_cluster(items)
        leaf_ids = [item["leaf_id"] for item in items]
        pages = list(set([item["page_number"] for item in items if item["page_number"] != "N/A"]))
        return cluster_id, summary, leaf_ids, pages, tokens

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_cluster, c_id, items) for c_id, items in cluster_map.items()]
        
        # Lấy kết quả và sắp xếp lại theo cluster_id cho đồng bộ
        results = [f.result() for f in futures]
        results.sort(key=lambda x: x[0])
        
        for res in results:
            parent_texts.append(res[1])
            parent_leaf_ids.append(res[2])
            parent_cited_pages.append(res[3])
            total_tokens_used += res[4]
    
    # embedding nút cha
    parent_vectors, parent_tokens = _embed_texts(parent_texts)
    total_tokens_used += parent_tokens

    # tạo global summary (level 2)
    global_summary, global_summary_tokens = _generate_global_summary(parent_texts, parent_cited_pages)
    total_tokens_used += global_summary_tokens
    
    global_vector, global_vector_tokens = _embed_texts([global_summary])
    total_tokens_used += global_vector_tokens

    # lưu vào trong qdrant
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
        vector=global_vector[0], 
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
    return {
        "total_leaf_nodes": len(leaf_ids),
        "total_parent_nodes": len(parent_leaf_ids),
        "total_tokens": total_tokens_used,
        "has_global_summary": True
    }