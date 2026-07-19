from __future__ import annotations
import logging
import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from config import (
    DEFAULT_TOP_K, 
    GOOGLE_API_KEYS, 
    QDRANT_COLLECTION_NAME,
    GITHUB_API_KEY,
    GITHUB_BASE_URL,
    GITHUB_FALLBACK_MODELS
)

logger = logging.getLogger(__name__)

# ==========================================
# 1. ĐỒNG BỘ EMBEDDING VỚI QDRANT
# ==========================================
if not GOOGLE_API_KEYS:
    raise ValueError("Lỗi cấu hình: Không tìm thấy GOOGLE_API_KEY.")

# Phải trùng khớp 100% với lúc Ingestion (Upload file)
query_embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2", 
    google_api_key=GOOGLE_API_KEYS[0],
    output_dimensionality=1536
)

def _retrieve_from_qdrant(query_vector: list[float], session_id: str, qdrant_client: QdrantClient, top_k: int) -> tuple[list[str], list[dict]]:
    """Lấy tài liệu thô từ Qdrant và dò ngược Citation"""
    results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]),
        limit=top_k,
        with_payload=True,
    )
    
    best_texts = []
    leaf_ids_to_fetch = set()
    direct_leaf_nodes = []
    
    for point in results.points:
        if not point.payload: continue
        text_content = point.payload.get("text_content", "")
        if not text_content: continue
        
        level = point.payload.get("level", 0)
        if level == 0:
            pages = [point.payload.get("page_number", "N/A")]
            page_str = ", ".join(map(str, pages))
            best_texts.append(f"Nội dung:\n{text_content}\nNguồn gốc: Trang {page_str}")
            direct_leaf_nodes.append({
                "page": point.payload.get("page_number", "N/A"),
                "text": text_content
            })
        else:
            pages = point.payload.get("cited_pages", [])
            page_str = ", ".join(map(str, pages)) if pages else "N/A"
            best_texts.append(f"Nội dung:\n{text_content}\nNguồn gốc: Trang {page_str}")
            
            l_ids = point.payload.get("leaf_ids", [])
            for lid in l_ids:
                if isinstance(lid, list):
                    leaf_ids_to_fetch.update(lid)
                else:
                    leaf_ids_to_fetch.add(lid)
                    
    # Lấy dữ liệu gốc từ Node Lá cho Citation
    citations = []
    for node in direct_leaf_nodes:
        if node["page"] != "N/A":
            citations.append(node)
            
    if leaf_ids_to_fetch:
        leaf_results = qdrant_client.retrieve(
            collection_name=QDRANT_COLLECTION_NAME,
            ids=list(leaf_ids_to_fetch),
            with_payload=True
        )
        for point in leaf_results:
            if point.payload:
                page = point.payload.get("page_number", "N/A")
                if page != "N/A":
                    citations.append({
                        "page": page,
                        "text": point.payload.get("text_content", "")
                    })
                    
    # Lọc trùng lặp Citation theo page
    unique_citations = {}
    for c in citations:
        page = c["page"]
        if page not in unique_citations:
            unique_citations[page] = c["text"]
        else:
            unique_citations[page] += "\n...\n" + c["text"]
            
    final_citations = [{"page": p, "text": t[:1000] + "..." if len(t) > 1000 else t} for p, t in unique_citations.items()]
    final_citations = sorted(final_citations, key=lambda x: int(x["page"]) if str(x["page"]).isdigit() else 9999)
    
    return best_texts, final_citations


# Hàm _filter_with_local_ai đã bị xóa bỏ để tránh rơi rụng dữ liệu.


async def rag_query_stream(user_question: str, session_id: str, qdrant_client: QdrantClient, top_k: int = DEFAULT_TOP_K) -> AsyncGenerator[str, None]:
    
    if not GITHUB_API_KEY:
        yield "Lỗi Backend: Chưa cấu hình GITHUB_API_KEY trong file config."
        return

    # Bước 1: Nhúng Vector Câu hỏi
    try:
        query_vector = query_embedder.embed_query(user_question)
    except Exception as e:
        yield f"Lỗi hệ thống khi phân tích câu hỏi: {str(e)}"
        return
    
    # Bước 2: Truy xuất diện rộng từ Qdrant
    fetch_k = top_k + 5 
    best_texts, citations = _retrieve_from_qdrant(query_vector, session_id, qdrant_client, fetch_k)
    
    import json
    citations_json = json.dumps(citations, ensure_ascii=False)
    
    if not best_texts:
        yield "Không tìm thấy tài liệu nào liên quan trong hệ thống lưu trữ."
        yield f"\n\n[CITATIONS] {citations_json}"
        return

    # Bước 3: Gộp trực tiếp tài liệu thô từ Qdrant
    raw_context = "\n\n[ĐOẠN TÀI LIỆU]\n".join(best_texts)

    # Bước 4: Prompt định hướng chất lượng cao cho LLM Đám Mây (GPT-4o, Claude)
    prompt = f"""Bạn là một trợ lý AI phân tích dữ liệu cấp cao.
Dưới đây là các đoạn dữ kiện thô được truy xuất trực tiếp từ tài liệu gốc. Mỗi dữ kiện đều được đánh dấu "Nguồn gốc: Trang X".
Hãy đọc toàn bộ chúng và nhặt ra thông tin để giải quyết câu hỏi của người dùng.

<DỮ KIỆN CỐT LÕI>
{raw_context}
</DỮ KIỆN CỐT LÕI>

<CÂU HỎI>
{user_question}
</CÂU HỎI>

YÊU CẦU ĐẦU RA:
1. Trình bày câu trả lời rõ ràng, mạch lạc, dễ hiểu bằng tiếng Việt chuẩn. Sử dụng cấu trúc danh sách hoặc in đậm để làm nổi bật ý chính.
2. TUYỆT ĐỐI trung thành với <DỮ KIỆN CỐT LÕI>. Không sử dụng kiến thức bên ngoài để phỏng đoán hay bịa đặt.
3. TRÍCH DẪN TẠI CHỖ (INLINE CITATIONS): BẮT BUỘC chèn thẻ trích dẫn dưới dạng Markdown link `[Trang X](#page-X)` ngay tại câu mà bạn lấy thông tin từ Trang X.
Ví dụ: "Hệ thống RAG giúp tìm kiếm ngữ nghĩa [Trang 1](#page-1)."
4. Nếu dữ kiện cung cấp không đủ để kết luận, hãy thông báo: "Dữ liệu hiện tại không đủ để trả lời chi tiết câu hỏi này" thay vì suy diễn.
"""

    # Bước 5: Gọi GitHub Models và Fallback
    for model_name in GITHUB_FALLBACK_MODELS:
        try:
            logger.info("hỏi đáp với model: %s...", model_name)
            
            # Cấu hình API OpenAI SDK trỏ thẳng sang cơ sở hạ tầng của GitHub
            llm = ChatOpenAI(
                model=model_name,
                api_key=GITHUB_API_KEY,
                base_url=GITHUB_BASE_URL,
                temperature=0.2, # Sáng tạo thấp để bám sát dữ kiện
                streaming=True,
                max_retries=0
            )
            
            async_generator = llm.astream(prompt)
            async for chunk in async_generator:
                if chunk.content: 
                    yield chunk.content
                    
            yield f"\n\n[CITATIONS] {citations_json}"
            return # Trả lời mượt mà xong thì kết thúc vòng lặp
            
        except Exception as e:
            logger.warning("Truy xuất qua %s thất bại: %s. Kích hoạt model dự phòng...", model_name, e)
            await asyncio.sleep(1)
            
    # Nếu tất cả Cloud Models đều lỗi
    logger.error("Đã thử toàn bộ GitHub Fallback Models nhưng đều thất bại.")
    yield "\n\n[Hệ thống API đám mây hiện đang quá tải hoặc gặp sự cố. Vui lòng thử lại sau.]"
    yield f"\n\n[CITATIONS] {citations_json}"