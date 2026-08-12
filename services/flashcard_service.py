import logging
import json
import random
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from config import (
    QDRANT_COLLECTION_NAME,
    GITHUB_API_KEY,
    GITHUB_BASE_URL,
    GITHUB_FALLBACK_MODELS,
    GOOGLE_API_KEYS,
    OLLAMA_SUMMARY_MODEL
)

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
            limit=1
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
                limit=10
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

    # 2. Xây dựng Prompt
    prompt = f"""Bạn là một chuyên gia sư phạm. Dựa vào nội dung tài liệu dưới đây, hãy trích xuất 10-15 khái niệm/định nghĩa quan trọng nhất dưới dạng Flashcard (Thẻ ghi nhớ).
    
QUY TẮC BẮT BUỘC:
1. Bạn PHẢI trả về ĐÚNG định dạng JSON mảng các object. Tuyệt đối không sinh thêm các câu hội thoại như "Đây là kết quả của bạn" hay đánh dấu markdown ```json. 
2. Mỗi object gồm 2 khóa:
   - "front": Khái niệm hoặc từ khóa cốt lõi (rất ngắn gọn, từ 1 đến 10 từ).
   - "back": Định nghĩa, giải thích chi tiết, dễ hiểu (từ 1 đến 3 câu).
3. Đảm bảo nội dung mặt sau KHÔNG chứa các ký tự lạ làm hỏng chuỗi JSON (như dấu nháy kép thừa). Chỉ dùng dấu nháy đơn ' cho nội dung bên trong nếu cần.

TÀI LIỆU CẦN XỬ LÝ:
{text_content}
"""

    # 3. Gọi AI Models (Thử Github GPT-4o trước, nếu fail thì qua Gemini, cuối cùng fallback Ollama)
    from langchain_openai import ChatOpenAI
    import json
    import re
    
    # Helper parser an toàn
    def parse_json_safely(text: str) -> List[Dict[str, str]]:
        # Tìm mảng JSON bên trong text
        match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            return json.loads(text)
        except Exception:
            return []

    # A. Cố gắng dùng GitHub API
    if GITHUB_API_KEY:
        for model_name in GITHUB_FALLBACK_MODELS:
            try:
                logger.info(f"Đang sinh Flashcard bằng Cloud AI: {model_name}...")
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=GITHUB_API_KEY,
                    base_url=GITHUB_BASE_URL,
                    temperature=0.2, 
                    max_retries=0
                )
                response = llm.invoke(prompt)
                parsed = parse_json_safely(response.content)
                if parsed:
                    return parsed
            except Exception as e:
                logger.warning(f"Truy xuất qua {model_name} tạo flashcard thất bại: {e}.")
                time.sleep(1)
                
    # B. Thử Google Gemini nếu GitHub sập
    if GOOGLE_API_KEYS:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = random.choice(GOOGLE_API_KEYS)
        try:
            logger.info("Đang sinh Flashcard bằng Google Gemini 2.5 Flash...")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.2,
                max_retries=0
            )
            response = llm.invoke(prompt)
            parsed = parse_json_safely(response.content)
            if parsed:
                return parsed
        except Exception as e:
            logger.warning(f"Google API tạo flashcard lỗi: {e}.")
            
    # C. Cùng đường thì xài Local Ollama
    from langchain_ollama import ChatOllama
    try:
        logger.info(f"Đang sinh Flashcard bằng Local Ollama: {OLLAMA_SUMMARY_MODEL}")
        llm = ChatOllama(
            model=OLLAMA_SUMMARY_MODEL, 
            temperature=0.1,      
            format="json",  # Ép ollama sinh JSON
            timeout=180.0
        )
        response = llm.invoke(prompt)
        # Vì format="json", đầu ra của Ollama thường trả về JSON object, cần check
        content = response.content
        parsed = parse_json_safely(content)
        if parsed:
            return parsed
        else:
            # Lỡ Ollama bọc trong dictionary kiểu {"flashcards": [...]}
            try:
                obj = json.loads(content)
                if isinstance(obj, dict):
                    for key, val in obj.items():
                        if isinstance(val, list):
                            return val
                elif isinstance(obj, list):
                    return obj
            except Exception:
                pass
                
    except Exception as e:
        logger.error(f"Lỗi Ollama sinh Flashcard: {e}")
        
    raise ValueError("Toàn bộ AI Model đều thất bại trong việc tạo Flashcard hợp lệ.")
