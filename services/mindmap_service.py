import logging
import json
import random
import time
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct

from config import (
    QDRANT_COLLECTION_NAME,
    GITHUB_API_KEY,
    GITHUB_BASE_URL,
    GITHUB_FALLBACK_MODELS,
    GOOGLE_API_KEYS,
    OLLAMA_SUMMARY_MODEL
)

logger = logging.getLogger(__name__)

def generate_mindmap(document_id: str, session_id: str, qdrant_client: QdrantClient) -> str:
    """
    Truy xuất mindmap từ bộ nhớ đệm (Qdrant). Nếu chưa có thì sinh bằng AI và lưu lại.
    Trả về chuỗi Markdown lồng nhau (tương thích markmap).
    """
    # 1. Kiểm tra xem mindmap đã được lưu (cache) trong Qdrant chưa
    try:
        cached_results = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                    FieldCondition(key="type", match=MatchValue(value="mindmap"))
                ]
            ),
            limit=1
        )
        points = cached_results[0]
        if points:
            logger.info("Mindmap đã tồn tại trong database. Lấy từ cache...")
            return points[0].payload.get("text_content", "")
    except Exception as e:
        logger.warning(f"Lỗi khi check cache mindmap: {e}")

    # 2. Nếu chưa có, lấy global_summary để tạo mindmap
    text_content = ""
    global_vector = None
    try:
        results = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                    FieldCondition(key="type", match=MatchValue(value="global_summary"))
                ]
            ),
            limit=1,
            with_vectors=True
        )
        
        points = results[0]
        if points:
            text_content = points[0].payload.get("text_content", "")
            global_vector = points[0].vector
    except Exception as e:
        logger.error(f"Lỗi khi lấy global_summary từ Qdrant: {e}")
        raise ValueError(f"Không thể truy xuất dữ liệu từ Qdrant: {e}")

    if not text_content:
        raise ValueError("Không tìm thấy thông tin tóm tắt tổng quan (global_summary) cho tài liệu này.")

    # 3. Chế tạo Prompt để ép AI xuất Markdown cho Mindmap
    prompt = f"""
Bạn là một chuyên gia tạo Sơ Đồ Tư Duy (Mindmap) chuyên nghiệp.
Dựa vào nội dung tóm tắt dưới đây, hãy trích xuất các ý chính và sắp xếp chúng thành một cấu trúc phân cấp (cây).

Nội Dung:
\"\"\"
{text_content}
\"\"\"

Yêu Cầu Chặt Chẽ:
1. TRẢ VỀ ĐÚNG ĐỊNH DẠNG MARKDOWN NESTED LIST (danh sách lồng nhau bằng markdown).
2. Dòng đầu tiên (Root) dùng thẻ H1: `# [Tên Chủ Đề Chính]`
3. Các nhánh con dùng H2 `##`, H3 `###` hoặc danh sách gạch ngang `- `.
4. KHÔNG bao gồm văn bản giải thích. KHÔNG sử dụng code block ```markdown ... ``` (nếu có dùng thì chỉ lấy phần ruột).
5. Thông tin phải cực kỳ ngắn gọn, dạng từ khóa hoặc câu rất ngắn (giống mindmap thực tế).

Ví dụ định dạng bạn phải trả về:
# Trí Tuệ Nhân Tạo
## Khái niệm
- Định nghĩa chung
- Lịch sử phát triển
## Ứng dụng
### Y tế
- Chẩn đoán hình ảnh
- Phân tích gen
### Giáo dục
- Học tập cá nhân hóa
"""

    mindmap_markdown = ""

    # A. Dùng GitHub Models trước (như flashcard_service)
    if GITHUB_API_KEY:
        from langchain_openai import ChatOpenAI
        for model_name in GITHUB_FALLBACK_MODELS:
            try:
                logger.info(f"Đang sinh Mindmap bằng GitHub Model: {model_name}...")
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=GITHUB_API_KEY,
                    base_url=GITHUB_BASE_URL,
                    temperature=0.1,
                    max_retries=0,
                    timeout=30.0
                )
                response = llm.invoke(prompt)
                mindmap_markdown = response.content
                break # Thành công thì dừng
            except Exception as e:
                logger.warning(f"Truy xuất qua {model_name} tạo mindmap thất bại: {e}.")
                time.sleep(1)

    # B. Thử Google Gemini
    if not mindmap_markdown and GOOGLE_API_KEYS:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = random.choice(GOOGLE_API_KEYS)
        try:
            logger.info("Đang sinh Mindmap bằng Google Gemini 2.5 Flash...")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.2,
                max_retries=0
            )
            response = llm.invoke(prompt)
            mindmap_markdown = response.content
        except Exception as e:
            logger.warning(f"Google API tạo mindmap lỗi: {e}.")
            
    # C. Thử Ollama
    if not mindmap_markdown:
        from langchain_ollama import ChatOllama
        try:
            logger.info(f"Đang sinh Mindmap bằng Local Ollama: {OLLAMA_SUMMARY_MODEL}")
            llm = ChatOllama(
                model=OLLAMA_SUMMARY_MODEL, 
                temperature=0.1,      
                timeout=180.0
            )
            response = llm.invoke(prompt)
            mindmap_markdown = response.content
        except Exception as e:
            logger.error(f"Lỗi Ollama sinh mindmap: {e}")

    if not mindmap_markdown:
        raise ValueError("Toàn bộ AI Model đều thất bại trong việc tạo Mindmap.")

    # Dọn dẹp markdown rác (ví dụ code block ```markdown ... ```)
    mindmap_markdown = mindmap_markdown.strip()
    if mindmap_markdown.startswith("```markdown"):
        mindmap_markdown = mindmap_markdown[11:]
    if mindmap_markdown.startswith("```"):
        mindmap_markdown = mindmap_markdown[3:]
    if mindmap_markdown.endswith("```"):
        mindmap_markdown = mindmap_markdown[:-3]
    mindmap_markdown = mindmap_markdown.strip()

    # 4. Lưu lại vào Qdrant để cache
    try:
        # Nếu global_summary không trả về vector (vì lý do nào đó), ta sinh một vector 0 giả
        if global_vector is None:
            # Qdrant cần biết vector dimension, tốt nhất là lấy vector từ query. 
            # Tuy nhiên with_vectors=True ở trên đã lấy vector rồi.
            pass
            
        if global_vector is not None:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=global_vector,
                payload={
                    "session_id": session_id,
                    "document_id": document_id,
                    "level": 3,
                    "text_content": mindmap_markdown,
                    "type": "mindmap"
                }
            )
            qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=[point])
            logger.info("Đã lưu cache Mindmap vào Qdrant thành công.")
    except Exception as e:
        logger.error(f"Không thể lưu mindmap vào Qdrant cache: {e}")

    return mindmap_markdown
