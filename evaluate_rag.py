import os
import io
import time
import json
import logging
import PyPDF2
from docx import Document
from rouge_score import rouge_scorer
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, MatchValue, Filter

# Các hàm từ project của bạn
from config import get_qdrant_client, QDRANT_COLLECTION_NAME, OPENROUTER_API_KEY
from services.raptor_ingestion import raptor_ingest
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_Evaluator")

# Cấu hình thư mục chứa tài liệu test
TEST_DOCS_DIR = "test_docs"

def extract_pdf_pages(file_path):
    pages = []
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                pages.append({"page_number": i + 1, "text": text})
    return pages

def evaluate_rouge(ai_summary, ref_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(ref_summary, ai_summary)
    return {
        "rouge1_f1": scores['rouge1'].fmeasure,
        "rouge2_f1": scores['rouge2'].fmeasure,
        "rougeL_f1": scores['rougeL'].fmeasure
    }

def run_evaluation():
    if not os.path.exists(TEST_DOCS_DIR):
        os.makedirs(TEST_DOCS_DIR)
        logger.error(f"Vui lòng copy 5-7 file PDF vào thư mục '{TEST_DOCS_DIR}' và chạy lại script.")
        return

    qdrant_client = get_qdrant_client()
    test_files = [f for f in os.listdir(TEST_DOCS_DIR) if f.endswith(".pdf") or f.endswith(".docx")]
    
    if not test_files:
        logger.error(f"Không tìm thấy file PDF/DOCX nào trong '{TEST_DOCS_DIR}'.")
        return

    results = []

    for filename in test_files:
        filepath = os.path.join(TEST_DOCS_DIR, filename)
        document_id = filename.rsplit(".", 1)[0].replace(" ", "_")
        session_id = f"test_session_{document_id}"
        
        logger.info(f"\n{'='*50}\nBắt đầu đánh giá tài liệu: {filename}\n{'='*50}")
        
        # 1. Trích xuất văn bản
        pages = []
        full_raw_text = ""
        if filename.endswith(".pdf"):
            pages = extract_pdf_pages(filepath)
            full_raw_text = "\n".join([p["text"] for p in pages])
        elif filename.endswith(".docx"):
            doc = Document(filepath)
            full_raw_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            if full_raw_text:
                pages.append({"page_number": 1, "text": full_raw_text})
        if not pages:
            logger.warning(f"File {filename} không có chữ.")
            continue
            
        # 2. Lập chỉ mục (Chunking & Embedding) & Tóm tắt (RAPTOR)
        start_time = time.time()
        logger.info("Đang chạy quá trình Ingestion (Chunking, Embedding, Clustering, Summarizing)...")
        ingest_result = raptor_ingest(pages, document_id, session_id, qdrant_client)
        process_time = time.time() - start_time
        
        leaf_nodes = ingest_result.get("total_leaf_nodes", 0)
        parent_nodes = ingest_result.get("total_parent_nodes", 0)
        total_vectors = leaf_nodes + parent_nodes + 1 # +1 for global summary
        
        logger.info(f"[Đánh giá Ingestion] Thời gian xử lý: {process_time:.2f} giây")
        logger.info(f"[Đánh giá Ingestion] Số lượng chunks (Leaves): {leaf_nodes}")
        logger.info(f"[Đánh giá Ingestion] Số lượng cụm (Parents): {parent_nodes}")
        
        # 3. Lấy Global Summary từ Qdrant
        res = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                FieldCondition(key="level", match=MatchValue(value=2)) # Root (Global Summary is level 2)
            ]),
            limit=1
        )
        
        global_summary = res[0][0].payload.get("text_content", "") if res[0] else ""
        
        # 4. Tính điểm ROUGE (So sánh với file mẫu)
        ref_filename = filename.rsplit(".", 1)[0] + "_ref.txt"
        ref_filepath = os.path.join(TEST_DOCS_DIR, "ref", ref_filename)
        
        rouge_scores = None
        if os.path.exists(ref_filepath):
            with open(ref_filepath, "r", encoding="utf-8") as f:
                ref_summary = f.read()
            rouge_scores = evaluate_rouge(global_summary, ref_summary)
            logger.info(f"[Đánh giá Tóm tắt] ROUGE-1: {rouge_scores['rouge1_f1']:.3f}, ROUGE-2: {rouge_scores['rouge2_f1']:.3f}, ROUGE-L: {rouge_scores['rougeL_f1']:.3f}")
        else:
            logger.warning(f"Không tìm thấy file tóm tắt mẫu '{ref_filepath}'. Bỏ qua tính điểm ROUGE.")
            
        # Lưu kết quả
        results.append({
            "filename": filename,
            "pages": len(pages),
            "process_time_seconds": round(process_time, 2),
            "total_chunks": leaf_nodes,
            "total_clusters": parent_nodes,
            "total_vectors": total_vectors,
            "rouge_scores": rouge_scores
        })

    # In báo cáo tổng hợp
    logger.info(f"\n{'*'*50}\nBÁO CÁO TỔNG HỢP (Chương 3)\n{'*'*50}")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    logger.info("Đã xuất báo cáo ra file 'evaluation_report.json'")
    logger.info("\nLƯU Ý CỦA AI: Để có điểm ROUGE, bạn hãy tạo file txt cùng tên (VD: tailieu_ref.txt) chứa bản tóm tắt mẫu (do người viết hoặc GPT-4 viết) vào thư mục test_docs nhé!")

if __name__ == "__main__":
    run_evaluation()
