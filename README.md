# AI_BE - Dịch vụ xử lý AI (Đồ án tốt nghiệp)

Đây là phần service chuyên xử lý các tác vụ AI/LLM cho hệ thống hỗ trợ học tập và tóm tắt tài liệu. Service này hoạt động độc lập và giao tiếp với phần Backend chính cũng như Frontend thông qua RESTful API.

## Kiến trúc tổng quan

Hệ thống được tách thành 3 repository riêng biệt để dễ quản lý và mở rộng:
* **AI_BE (Repo này):** Xử lý bóc tách tài liệu, tạo vector embedding, tìm kiếm ngữ nghĩa (RAG) và chạy luồng tóm tắt phân cấp (RAPTOR) kết hợp hồ sơ người học.
* **DATN-BE:** Backend chính (Java / Spring Boot), quản lý người dùng, phân quyền, lưu trữ dữ liệu quan hệ và làm cầu nối gọi API sang `AI_BE`.
* **DATN-FE:** Giao diện người dùng (ReactJS).

Link 2 repo còn lại:
* Backend : https://github.com/vthai22825/DATN-BE
* Frontend: https://github.com/vthai22825/DATN-FE

## Công nghệ sử dụng

* **Ngôn ngữ & Framework:** Python, FastAPI, Uvicorn
* **Vector Database:** Qdrant
* **AI / LLM:** OpenRouter API (Gemini), BGE-M3 (Embedding)
* **Giao tiếp:** RESTful API (JSON)

## Cấu trúc thư mục
```text
AI_BE/
├── app/
│   ├── api/          # Định nghĩa các endpoint RESTful API
│   ├── core/         # Cấu hình hệ thống, biến môi trường
│   ├── models/       # Schema dữ liệu (Pydantic)
│   ├── services/     # Logic xử lý chính (RAG, tóm tắt phân cấp, embedding)
│   └── utils/        # Các hàm bổ trợ (cắt văn bản, đọc file)
├── main.py           # File khởi chạy ứng dụng
├── requirements.txt  # Danh sách thư viện cần cài đặt
└── .env.example      # Mẫu cấu hình biến môi trường
```

## Các chức năng chính

Dưới đây là các tính năng cốt lõi được xây dựng và tích hợp hoàn chỉnh trong hệ thống:

### 1. Xử lý tài liệu đa định dạng (Document Ingestion)
* **Hỗ trợ PDF & DOCX:** Xử lý file PDF (bóc tách theo trang vật lý) và DOCX (xử lý luồng văn bản liên tục bằng `RecursiveCharacterTextSplitter`).
* **Lưu trữ đám mây:** Tự động đẩy tệp gốc lên Cloudflare R2 để lưu trữ an toàn và tối ưu tốc độ truy xuất.

### 2. Kiến trúc tóm tắt phân cấp (RAPTOR Pipeline)
* **Phân mảnh & Gom cụm:** Chia văn bản thành Chunk Level 0, dùng thuật toán K-Means Clustering để gom nhóm cụm chủ đề và sinh Node Level 1.
* **Tóm tắt toàn cục (Global Summary):** Tự động tạo bản tóm tắt toàn bộ tài liệu chạy ngầm ngay trong pha Ingestion (`Background Ingestion`), cache trực tiếp vào hệ thống để gọi lại tức thời.
* **Lưu trữ Vector:** Toàn bộ cấu trúc đa tầng được nhúng qua mô hình BGE-M3 và lưu trữ vào Qdrant Vector DB (tổng cộng gần 500 vectors trong các bài test thực nghiệm).

### 3. Hỏi đáp ngữ cảnh (RAG Chatbot & Citations)
* **Truy xuất thông minh:** Tìm kiếm độ tương đồng Cosine trên Qdrant để lọc các đoạn ngữ cảnh sát với câu hỏi nhất.
* **Chống ảo giác (Zero-Hallucination):** Dùng mô hình Gemini 3.7 Flash với tham số `temperature = 0.0` để đảm bảo AI trả lời đúng 100% dựa trên tài liệu, không tự bịa thông tin.
* **Trích dẫn nguồn:** Mỗi câu trả lời đều đính kèm số trang/nguồn tài liệu gốc để người dùng dễ kiểm chứng.

### 4. Công cụ hỗ trợ học tập
* **Sơ đồ tư duy (Mind Map):** Hiển thị bản tóm tắt toàn cục dạng cây phân cấp (từ RAPTOR), giúp nắm bắt tổng quan tài liệu nhanh chóng.
* **Flashcards tự động:** Trích xuất các thuật ngữ và ý chính từ tài liệu để tạo bộ thẻ lật ôn tập kiến thức.
