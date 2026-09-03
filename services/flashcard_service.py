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


def generate_flashcards(document_id: str, session_id: str, number_of_cards: str, difficulty: str, topic: str, qdrant_client: QdrantClient) -> List[Dict[str, str]]:
    """
    Truy xuất nội dung tóm tắt của tài liệu từ Qdrant và dùng AI để sinh ra Flashcards.
    Có cơ chế Cache để không gọi AI nhiều lần.
    """
    # (Đã gỡ bỏ logic check Cache Qdrant, hệ thống luôn sinh thẻ mới theo mô hình NotebookLM)

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
            with_vectors=False
        )
        
        points = results[0]
        if points:
            text_content = points[0].payload.get("text_content", "")
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
                with_vectors=False
            )
            page_points = page_results[0]
            if not page_points:
                raise ValueError("Không tìm thấy nội dung tài liệu để tạo Flashcard.")
            
            text_content = "\n\n".join([p.payload.get("text_content", "") for p in page_points])
            
    except Exception as e:
        logger.error(f"Lỗi khi đọc Qdrant tạo Flashcard: {e}")
        raise ValueError(f"Không thể truy xuất dữ liệu từ DB: {str(e)}")
        
    if not text_content.strip():
        raise ValueError("Nội dung tài liệu trống.")

    # 2. Xây dựng cấu hình Prompt dựa vào yêu cầu người dùng
    number_inst = "10 đến 15 thẻ"
    if number_of_cards == "Ít hơn":
        number_inst = "5 đến 7 thẻ"
    elif number_of_cards == "Nhiều hơn":
        number_inst = "20 đến 25 thẻ"
        
    diff_inst = "Tập trung vào các khái niệm cốt lõi, có giải thích chi tiết và ví dụ minh họa nếu cần."
    if difficulty == "Dễ":
        diff_inst = "Chỉ tập trung vào các từ vựng, khái niệm cơ bản nhất. Mặt sau giải thích cực kỳ ngắn gọn, dễ hiểu cho người mới bắt đầu."
    elif difficulty == "Khó":
        diff_inst = "Đòi hỏi tư duy suy luận. Tạo ra các câu hỏi dạng so sánh, phân tích chuyên sâu, hoặc tình huống phức tạp có trong tài liệu."
        
    topic_inst = "Quét toàn bộ tài liệu để tìm các ý chính quan trọng nhất."
    if topic and topic.strip():
        topic_inst = f"CHỈ TẬP TRUNG TẠO THẺ CHO CHỦ ĐỀ SAU: '{topic.strip()}'. Bỏ qua các nội dung không liên quan."

    prompt = f"""Bạn là một chuyên gia sư phạm. Dựa vào nội dung tài liệu dưới đây, hãy trích xuất các khái niệm quan trọng nhất dưới dạng Flashcard (Thẻ ghi nhớ).
    
YÊU CẦU CỤ THỂ:
- Số lượng thẻ: Bạn cần sinh ra khoảng {number_inst}.
- Độ khó và Nội dung: {diff_inst}
- Chủ đề tập trung: {topic_inst}

QUY TẮC BẮT BUỘC:
1. Bạn PHẢI trả về ĐÚNG 1 ĐỐI TƯỢNG JSON (JSON Object) có duy nhất 1 key là "flashcards", giá trị là một mảng các đối tượng.
2. Mỗi object trong mảng gồm đúng 2 khóa:
   - "front": Khái niệm, từ khóa cốt lõi, hoặc câu hỏi (rất ngắn gọn, từ 1 đến 10 từ).
   - "back": Định nghĩa, giải thích chi tiết, hoặc câu trả lời (từ 1 đến 3 câu).

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

    # 4. (Đã gỡ bỏ logic lưu Cache vào Qdrant. Java BE MySQL sẽ đảm nhận việc lưu trữ)

    return final_flashcards, total_tokens
