import logging
import json
import time
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct

from config import (
    QDRANT_COLLECTION_NAME,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_CHAT_MODEL
)
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

def generate_mindmap(document_id: str, session_id: str, qdrant_client: QdrantClient) -> tuple[str, int]:
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
            return points[0].payload.get("text_content", ""), 0
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
    total_tokens = 0

    if OPENROUTER_API_KEY:
        try:
            logger.info(f"Đang sinh Mindmap bằng OpenRouter Model: {OPENROUTER_CHAT_MODEL}...")
            llm = ChatOpenAI(
                model=OPENROUTER_CHAT_MODEL,
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                temperature=0.1,
                max_retries=1,
                timeout=60.0
            )
            response = llm.invoke(prompt)
            mindmap_markdown = response.content
            total_tokens = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
        except Exception as e:
            logger.warning(f"Truy xuất OpenRouter tạo mindmap thất bại: {e}.")

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

    return mindmap_markdown, total_tokens
