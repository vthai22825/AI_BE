import logging
import json
import time
import uuid
import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field
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

# Cấu trúc đầu ra mong muốn cho LangChain (Structured Output)
class FlashcardOutput(BaseModel):
    front: str = Field(description="Mặt trước của thẻ: Khái niệm, từ khóa hoặc câu hỏi ngắn gọn (tối đa 10 từ).")
    back: str = Field(description="Mặt sau của thẻ: Định nghĩa, ý nghĩa hoặc câu trả lời chi tiết, dễ hiểu.")

class FlashcardListOutput(BaseModel):
    flashcards: List[FlashcardOutput] = Field(description="Danh sách gồm 10 đến 15 thẻ flashcard.")


def generate_flashcards(document_id: str, session_id: str, qdrant_client: QdrantClient) -> List[Dict[str, str]]:
    """
    Truy xuất nội dung tóm tắt của tài liệu từ Qdrant và dùng AI để sinh ra Flashcards.
    Có cơ chế Cache để không gọi AI nhiều lần.
    """
    # 0. Kiểm tra xem flashcards đã được cache chưa
    try:
        cached_results = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                    FieldCondition(key="type", match=MatchValue(value="flashcard"))
                ]
            ),
            limit=1
        )
        points = cached_results[0]
        if points:
            logger.info("Flashcards đã tồn tại trong database. Lấy từ cache...")
            cached_json_str = points[0].payload.get("text_content", "[]")
            try:
                return json.loads(cached_json_str), 0
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Lỗi khi check cache flashcards: {e}")

    # 1. Lấy nội dung từ Qdrant (ưu tiên global_summary)
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
        else:
            # Nếu không có global summary, lôi tạm vài chunk ở level 0 ra gộp lại
            page_results = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                        FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                        FieldCondition(key="level", match=MatchValue(value=0))
                    ]
                ),
                limit=10,
                with_vectors=True
            )
            page_points = page_results[0]
            if not page_points:
                raise ValueError("Không tìm thấy nội dung tài liệu để tạo Flashcard.")
            
            text_content = "\n\n".join([p.payload.get("text_content", "") for p in page_points])
            global_vector = page_points[0].vector
            
    except Exception as e:
        logger.error(f"Lỗi khi đọc Qdrant tạo Flashcard: {e}")
        raise ValueError(f"Không thể truy xuất dữ liệu từ DB: {str(e)}")
        
    if not text_content.strip():
        raise ValueError("Nội dung tài liệu trống.")

    # 2. Xây dựng Prompt
    prompt = f"""Bạn là một chuyên gia sư phạm. Dựa vào nội dung tài liệu dưới đây, hãy trích xuất 10-15 khái niệm/định nghĩa quan trọng nhất dưới dạng Flashcard (Thẻ ghi nhớ).
    
QUY TẮC BẮT BUỘC:
1. Bạn PHẢI trả về ĐÚNG 1 ĐỐI TƯỢNG JSON (JSON Object) có duy nhất 1 key là "flashcards", giá trị là một mảng các đối tượng.
2. Mỗi object trong mảng gồm đúng 2 khóa:
   - "front": Khái niệm hoặc từ khóa cốt lõi (rất ngắn gọn, từ 1 đến 10 từ).
   - "back": Định nghĩa, giải thích chi tiết, dễ hiểu (từ 1 đến 3 câu).

TÀI LIỆU CẦN XỬ LÝ:
{text_content}
"""

    # Helper parser an toàn
    def parse_json_safely(text: str) -> List[Dict[str, str]]:
        match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            return json.loads(text)
        except Exception:
            return []

    final_flashcards = []
    total_tokens = 0
    
    if OPENROUTER_API_KEY:
        try:
            logger.info(f"Đang sinh Flashcard bằng OpenRouter AI: {OPENROUTER_CHAT_MODEL}...")
            llm = ChatOpenAI(
                model=OPENROUTER_CHAT_MODEL,
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                temperature=0.2, 
                max_retries=1,
                # Ép kiểu JSON_OBJECT cho các mô hình hỗ trợ
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            response = llm.invoke(prompt)
            total_tokens = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
            
            # Giải mã JSON an toàn
            try:
                res_obj = json.loads(response.content)
                if "flashcards" in res_obj:
                    final_flashcards = res_obj["flashcards"]
                else:
                    final_flashcards = parse_json_safely(response.content)
            except Exception:
                final_flashcards = parse_json_safely(response.content)
                
        except Exception as e:
            logger.warning(f"Truy xuất qua OpenRouter tạo flashcard thất bại: {e}.")
            
    if not final_flashcards:
        raise ValueError("AI Model thất bại trong việc tạo Flashcard hợp lệ.")

    # 4. Lưu Cache vào Qdrant
    try:
        if global_vector is not None:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=global_vector,
                payload={
                    "session_id": session_id,
                    "document_id": document_id,
                    "level": 3,
                    "text_content": json.dumps(final_flashcards, ensure_ascii=False),
                    "type": "flashcard"
                }
            )
            qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=[point])
            logger.info("Đã lưu cache Flashcards vào Qdrant thành công.")
    except Exception as e:
        logger.error(f"Không thể lưu flashcard vào Qdrant cache: {e}")

    return final_flashcards, total_tokens
