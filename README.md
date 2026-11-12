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

## Các chức năng chính

- **Xử lý và lập chỉ mục tài liệu (Document Ingestion):** Tiếp nhận file tài liệu từ `DATN-BE` (qua URL hoặc luồng dữ liệu), bóc tách nội dung văn bản, chia nhỏ thành các đoạn (chunking) và mã hóa thành vector bằng mô hình `BGE-M3` trước khi lưu trữ vào `Qdrant`.
- **Tóm tắt tài liệu phân cấp (Hierarchical Summarization):** Xây dựng cây tóm tắt nhiều tầng theo kiến trúc RAPTOR (gom cụm các đoạn văn bản có ngữ nghĩa tương đồng và tóm tắt dần từ chi tiết lên tổng quan), giúp xử lý hiệu quả các tài liệu dài mà không bị mất ngữ cảnh.
- **Tóm tắt cá nhân hóa theo người học:** Kết hợp cấu trúc cây tài liệu với trạng thái và lịch sử học tập của từng người dùng (do `DATN-BE` truyền sang) để điều chỉnh độ sâu, trọng tâm và cách diễn đạt của bản tóm tắt.
- **Hỏi đáp trên tài liệu (RAG Q&A):** Truy xuất các đoạn văn bản và node tóm tắt liên quan nhất từ `Qdrant` dựa trên câu hỏi của người dùng, sau đó tổng hợp ngữ cảnh và gọi LLM (thông qua OpenRouter API) để sinh câu trả lời bám sát nội dung tài liệu.
- **Cung cấp RESTful API giao tiếp liên dịch vụ:** Đóng gói toàn bộ luồng xử lý thành các endpoint chuẩn JSON (tài liệu hóa tự động qua Swagger UI) để `DATN-BE` gọi bất đồng bộ hoặc đồng bộ tùy theo tác vụ.
