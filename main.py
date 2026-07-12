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
    workspace_id: str = Form("ws-001"),
    document_id: str = Form("doc-001")
)-> DocumentProcessResponse:
    try:
        #đọc và trích dẫn file
        content = file.file.read()
        text = ""
        if file.filename and file.filename.lower().endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            extracted = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
            text = "\n\n".join(extracted)
        else: 
            text = content.decode("utf-8")
        if not text.strip():
            raise ValueError("file không có chữ")
        
        #chuyển văn bản thô sau khi lấy được xuống service chạy ngầm
        qdrant_client = get_qdrant_client()
        background_tasks.add_task(raptor_ingest, text, document_id, workspace_id, qdrant_client)

        return DocumentProcessResponse(
            message=f"Đang xử lý file {file.filename} ngầm",
            document_id=document_id,
            total_leaf_nodes=0,
            total_parent_node=0
        )
    except Exception as e:
        logger.error("Lỗi upload: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{document_id}/summary", tags=["Document"])
def get_document_summary(document_id: str, workspace_id: str = "ws-001"):
    try:
        qdrant_client = get_qdrant_client()
        
        results = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    FieldCondition(key="workspace_id", match=MatchValue(value=workspace_id)),
                    FieldCondition(key="type", match=MatchValue(value="global_summary"))
                ]
            ),
            limit=1
        )
        
        points = results[0]
        if not points:
            return {"status": "processing", "message": "Tài liệu đang được tóm tắt hoặc chưa hoàn thành."}
            
        summary_text = points[0].payload.get("text_content", "")
        return {"status": "success", "summary": summary_text}
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy summary: {str(e)}")
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
            try:
                # Gọi hàm rag_query_stream từ file ai_query.py
                async for token in rag_query_stream(request.user_question, request.workspace_id, qdrant_client):
                    # Format chuẩn SSE: phải có chữ "data: " và kết thúc bằng "\n\n"
                    yield f"data: {token}\n\n"
                
                # Báo hiệu cho Frontend biết luồng đã kết thúc
                yield "data: [DONE]\n\n"
                
            except Exception as stream_error:
                logger.error(f"Lỗi trong quá trình stream: {stream_error}")
                # Gửi thông báo lỗi về cho Frontend nếu AI sập giữa chừng
                yield f"data: [ERROR] Lỗi sinh chữ: {str(stream_error)}\n\n"

        # Trả về Response với media_type="text/event-stream"
        return StreamingResponse(content=event_generator(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Lỗi khởi tạo hệ thống stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
