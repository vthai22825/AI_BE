from pydantic import BaseModel, Field

class DocumentProcessResponse(BaseModel):
    status: str = "success"
    message: str
    document_id: str
    total_leaf_nodes: int
    total_parent_node: int

class ChatStreamRequest(BaseModel):
    session_id: str
    user_question: str

class FlashcardGenerateRequest(BaseModel):
    session_id: str
    number_of_cards: str = "Tiêu chuẩn"
    difficulty: str = "Trung bình"
    topic: str = ""

class QuizGenerateRequest(BaseModel):
    session_id: str
    number_of_questions: str = "Tiêu chuẩn"
    difficulty: str = "Trung bình"
    topic: str = ""