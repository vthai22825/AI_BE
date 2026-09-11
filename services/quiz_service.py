import logging
import json
import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from config import (
    QDRANT_COLLECTION_NAME,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_CHAT_MODEL
)
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Cấu trúc đầu ra mong muốn cho LangChain (Structured Output)
class QuizOption(BaseModel):
    id: str = Field(description="ID của đáp án (ví dụ: A, B, C, D).")
    text: str = Field(description="Nội dung của đáp án.")

class QuizQuestionOutput(BaseModel):
    question: str = Field(description="Nội dung câu hỏi trắc nghiệm.")
    options: List[QuizOption] = Field(description="Danh sách đúng 4 đáp án A, B, C, D.")
    correctAnswer: str = Field(description="ID của đáp án đúng (ví dụ: A).")
    hint: str = Field(description="Gợi ý ngắn gọn giúp người học suy luận ra đáp án.")
    explanation: str = Field(description="Giải thích chi tiết vì sao đáp án đó đúng.")

class QuizListOutput(BaseModel):
    quizzes: List[QuizQuestionOutput] = Field(description="Danh sách các câu hỏi trắc nghiệm.")


def generate_quiz(document_id: str, session_id: str, number_of_questions: str, difficulty: str, topic: str, qdrant_client: QdrantClient) -> tuple[List[Dict[str, Any]], int]:
    """
    Truy xuất nội dung tóm tắt của tài liệu từ Qdrant và dùng AI để sinh ra Quiz trắc nghiệm.
    """
    # 1. Lấy nội dung từ Qdrant (ưu tiên global_summary)
    text_content = ""
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
                raise ValueError("Không tìm thấy nội dung tài liệu để tạo Quiz.")
            
            text_content = "\n\n".join([p.payload.get("text_content", "") for p in page_points])
            
    except Exception as e:
        logger.error(f"Lỗi khi đọc Qdrant tạo Quiz: {e}")
        raise ValueError(f"Không thể truy xuất dữ liệu từ DB: {str(e)}")
        
    if not text_content.strip():
        raise ValueError("Nội dung tài liệu trống.")

    # 2. Xây dựng cấu hình Prompt dựa vào yêu cầu người dùng
    number_inst = "sinh chính xác 10 câu hỏi"
    if number_of_questions == "Ít hơn":
        number_inst = "sinh chính xác 5 câu hỏi"
    elif number_of_questions == "Nhiều hơn":
        number_inst = "sinh chính xác 20 câu hỏi"
        
    diff_inst = "Yêu cầu hiểu mối liên hệ giữa các khái niệm. Đưa ra các tình huống nhỏ để người học chọn cách xử lý đúng dựa trên tài liệu. Các đáp án nhiễu có sự tương đồng nhất định, đòi hỏi đọc kỹ tài liệu."
    if difficulty == "Dễ":
        diff_inst = "Hỏi trực tiếp về các khái niệm cơ bản, định nghĩa, hoặc số liệu rõ ràng trong tài liệu. Các đáp án nhiễu (sai) dễ dàng bị loại trừ."
    elif difficulty == "Khó":
        diff_inst = "Câu hỏi mang tính chất suy luận logic, so sánh, hoặc đánh giá sâu một vấn đề phức tạp. Các đáp án nhiễu được thiết kế cực kỳ tinh vi (bẫy), thoạt nhìn có vẻ đúng nhưng sai ở bản chất."
        
    topic_inst = "Quét toàn bộ tài liệu để tìm các ý chính quan trọng nhất."
    if topic and topic.strip():
        topic_inst = f"CHỈ TẬP TRUNG TẠO CÂU HỎI CHO CHỦ ĐỀ SAU: '{topic.strip()}'. Bỏ qua các nội dung không liên quan."

    prompt = f"""Bạn là một chuyên gia sư phạm. Dựa vào nội dung tài liệu dưới đây, hãy trích xuất các kiến thức quan trọng và tạo thành bộ câu hỏi Trắc nghiệm (Quiz).
    
YÊU CẦU CỤ THỂ:
- Số lượng: Bạn cần {number_inst}.
- Độ khó và Nội dung: {diff_inst}
- Chủ đề tập trung: {topic_inst}

QUY TẮC BẮT BUỘC:
1. Bạn PHẢI trả về ĐÚNG 1 ĐỐI TƯỢNG JSON (JSON Object) có duy nhất 1 key là "quizzes", giá trị là một mảng các đối tượng câu hỏi.
2. Mỗi câu hỏi trong mảng gồm:
   - "question": Nội dung câu hỏi.
   - "options": Mảng đúng 4 đáp án. Mỗi đáp án là object có "id" (A, B, C, D) và "text".
   - "correctAnswer": ID của đáp án đúng (ví dụ: "A").
   - "hint": Một gợi ý nhỏ giúp suy luận.
   - "explanation": Giải thích chi tiết tại sao đáp án đó đúng.

TÀI LIỆU CẦN XỬ LÝ:
{text_content}
"""

    # Helper parser an toàn
    def parse_json_safely(text: str) -> List[Dict[str, Any]]:
        match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            return json.loads(text)
        except Exception:
            return []

    final_quizzes = []
    total_tokens = 0
    
    if OPENROUTER_API_KEY:
        try:
            logger.info(f"Đang sinh Quiz bằng OpenRouter AI: {OPENROUTER_CHAT_MODEL}...")
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
                if "quizzes" in res_obj:
                    final_quizzes = res_obj["quizzes"]
                else:
                    final_quizzes = parse_json_safely(response.content)
            except Exception:
                final_quizzes = parse_json_safely(response.content)
                
        except Exception as e:
            logger.warning(f"Truy xuất qua OpenRouter tạo quiz thất bại: {e}.")
            
    if not final_quizzes:
        raise ValueError("AI Model thất bại trong việc tạo Quiz hợp lệ.")

    return final_quizzes, total_tokens
