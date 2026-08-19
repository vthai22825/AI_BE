from __future__ import annotations
import logging
import io
import PyPDF2
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from models.schemas import DocumentProcessResponse, ChatStreamRequest

from services.ai_query import rag_query_stream
from services.raptor_ingestion import raptor_ingest

from config import get_qdrant_client, QDRANT_COLLECTION_NAME
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
import time
from services.logger_service import log_event_async


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname) -7s | %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI(title="AI backend - RAPTOR and LLMLingua RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/documents/upload", response_model=DocumentProcessResponse, tags=["Document"])
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    document_id: str = Form(...)
)-> DocumentProcessResponse:
    try:
        #đọc và trích dẫn file
        content = file.file.read()
        pages = []
        if file.filename and file.filename.lower().endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            for i, page in enumerate(pdf_reader.pages):
                extracted_text = page.extract_text()
                if extracted_text:
                    pages.append({"page_number": i + 1, "text": extracted_text})
        elif file.filename and file.filename.lower().endswith((".doc", ".docx")):
            try:
                from docx import Document as DocxDocument
                doc = DocxDocument(io.BytesIO(content))
                full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                if full_text:
                    pages.append({"page_number": 1, "text": full_text})
            except Exception as e:
                raise ValueError(f"Không thể đọc file Word. Vui lòng đảm bảo đây là file .docx hợp lệ. (Lỗi: {str(e)})")
        elif file.filename and file.filename.lower().endswith(".txt"): 
            try:
                text = content.decode("utf-8")
                if text.strip():
                    pages.append({"page_number": 1, "text": text})
            except UnicodeDecodeError:
                raise ValueError("Không thể đọc file TXT do sai bảng mã. Vui lòng lưu file bằng UTF-8.")
        else:
            raise ValueError("Định dạng file không được hỗ trợ. Chỉ hỗ trợ .pdf, .docx, .txt")
                
        if not pages:
            raise ValueError("file không có chữ")
        
        # Chạy đồng bộ trực tiếp (chờ xử lý xong mới trả về)
        qdrant_client = get_qdrant_client()
        start_time = time.time()
        result = raptor_ingest(pages, document_id, session_id, qdrant_client)
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Token cost from result if available, else estimate
        token_cost = result.get("total_tokens", 0)
        user_id = session_id.split('_')[0] if '_' in session_id else "unknown" # Temporary fallback if no user email
        log_event_async("INFO", user_id, "Upload Document (RAPTOR)", latency_ms, token_cost)

        return DocumentProcessResponse(
            message=f"Đã xử lý file {file.filename} thành công",
            document_id=document_id,
            total_leaf_nodes=result.get("total_leaf_nodes", 0),
            total_parent_node=result.get("total_parent_nodes", 0)
        )
    except Exception as e:
        import traceback
        logger.error(f"Lỗi upload: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{document_id}/summary", tags=["Document"])
def get_document_summary(document_id: str, session_id: str):
    try:
        qdrant_client = get_qdrant_client()
        
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
        if not points:
            return {"status": "processing", "message": "Tài liệu đang được tóm tắt hoặc chưa hoàn thành."}
            
        summary_text = points[0].payload.get("text_content", "")
        cited_pages = points[0].payload.get("cited_pages", [])
        return {"status": "success", "summary": summary_text, "cited_pages": cited_pages}
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{document_id}/flashcards", tags=["Document"])
def get_flashcards(document_id: str, session_id: str):
    from services.flashcard_service import generate_flashcards
    try:
        qdrant_client = get_qdrant_client()
        start_time = time.time()
        flashcards, token_cost = generate_flashcards(document_id, session_id, qdrant_client)
        latency_ms = int((time.time() - start_time) * 1000)
        user_id = session_id.split('_')[0] if '_' in session_id else "unknown"
        log_event_async("INFO", user_id, "Generate Flashcard", latency_ms, token_cost)
        return {"status": "success", "flashcards": flashcards}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi API Flashcard: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống khi sinh flashcard: {str(e)}")

@app.get("/api/documents/{document_id}/mindmap", tags=["Document"])
def get_mindmap(document_id: str, session_id: str):
    from services.mindmap_service import generate_mindmap
    try:
        qdrant_client = get_qdrant_client()
        start_time = time.time()
        mindmap_markdown, token_cost = generate_mindmap(document_id, session_id, qdrant_client)
        latency_ms = int((time.time() - start_time) * 1000)
        user_id = session_id.split('_')[0] if '_' in session_id else "unknown"
        log_event_async("INFO", user_id, "Generate Mindmap", latency_ms, token_cost)
        return {"status": "success", "mindmap": mindmap_markdown}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi API Mindmap: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống khi sinh mindmap: {str(e)}")

@app.get("/api/documents/{document_id}/pages/{page_number}", tags=["Document"])
def get_document_page(document_id: str, page_number: int, session_id: str):
    try:
        qdrant_client = get_qdrant_client()
        results = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                    FieldCondition(key="page_number", match=MatchValue(value=page_number)),
                    FieldCondition(key="level", match=MatchValue(value=0))
                ]
            ),
            limit=50
        )
        
        points = results[0]
        if not points:
            return {"page_number": page_number, "text": "Không tìm thấy nội dung cho trang này."}
            
        # Gộp tất cả các chunk thuộc trang này
        text = "\n\n".join([p.payload.get("text_content", "") for p in points])
        return {"page_number": page_number, "text": text}
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy nội dung trang {page_number}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
    """
    API Hỏi Đáp (Stream). 
    Trả về từng đoạn văn bản (chunks) ngay khi AI đang suy nghĩ (Server-Sent Events).
    """
    try:
        qdrant_client = get_qdrant_client()
        
        async def event_generator():
            start_time = time.time()
            total_tokens = 0
            try:
                # Gọi hàm rag_query_stream từ file ai_query.py
                async for token in rag_query_stream(request.user_question, request.session_id, qdrant_client):
                    if token.startswith("\n\n[USAGE_TOKENS]"):
                        total_tokens = int(token.split("] ")[1])
                        continue
                        
                    # Format chuẩn SSE: phải có chữ "data: " và kết thúc bằng "\n\n"
                    # Xử lý xuống dòng trong token: thay \n bằng \ndata: để giữ đúng chuẩn SSE nhiều dòng
                    formatted_token = token.replace("\n", "\ndata: ")
                    yield f"data: {formatted_token}\n\n"
                
                # Báo hiệu cho Frontend biết luồng đã kết thúc
                yield "data: [DONE]\n\n"
                
                latency_ms = int((time.time() - start_time) * 1000)
                user_id = request.session_id.split('_')[0] if '_' in request.session_id else "unknown"
                log_event_async("INFO", user_id, "Chat Query", latency_ms, total_tokens)
                
            except Exception as stream_error:
                latency_ms = int((time.time() - start_time) * 1000)
                user_id = request.session_id.split('_')[0] if '_' in request.session_id else "unknown"
                log_event_async("ERROR", user_id, "Chat Query Failed", latency_ms, 0)
                logger.error(f"Lỗi trong quá trình stream: {stream_error}")
                # Gửi thông báo lỗi về cho Frontend nếu AI sập giữa chừng
                yield f"data: [ERROR] Lỗi sinh chữ: {str(stream_error)}\n\n"

        # Trả về Response với media_type="text/event-stream"
        return StreamingResponse(content=event_generator(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Lỗi khởi tạo hệ thống stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/api/documents/{document_id}", tags=["Document"])
def delete_document(document_id: str):
    """
    Xóa toàn bộ các vectors (chunks) của một Document cụ thể trên Qdrant.
    """
    try:
        qdrant_client = get_qdrant_client()
        # Dùng qdrant_client.delete với points_selector là một Filter
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )
        logger.info(f"Đã xóa toàn bộ vectors của document_id={document_id} trên Qdrant.")
        return {"status": "success", "message": f"Đã xóa dữ liệu của tài liệu {document_id}"}
    except Exception as e:
        logger.error(f"Lỗi khi xóa document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
