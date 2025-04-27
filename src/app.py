import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import process_pdf, VietnameseEmbedder, FAISSVectorStore, retriever_agent, get_rag_agent
from persistent_memory import PersistentMemory
import shutil
from typing import List
import logging
import sqlite3

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn lưu PDF
PDF_DIR = "../data/pdfs"
PDF_PATH = os.path.join(PDF_DIR, "uploaded.pdf")
os.makedirs(PDF_DIR, exist_ok=True)

# Khởi tạo global variables
memory = PersistentMemory(db_path="../data/memory.db", max_history=25)
embedder = None
vector_store = None
rag_agent = None


class QueryRequest(BaseModel):
    query: str


class HistoryItem(BaseModel):
    query: str
    response: str
    timestamp: str


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global embedder, vector_store, rag_agent
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File phải là PDF")

    try:
        # Lưu file PDF
        with open(PDF_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Xử lý PDF
        documents = process_pdf(PDF_PATH)
        logger.info(f"Đã xử lý PDF, tạo {len(documents)} chunks")

        # Khởi tạo embedder và vector store
        embedder = VietnameseEmbedder()
        vector_store = FAISSVectorStore(documents, embedder)
        rag_agent = get_rag_agent()

        return {"message": "PDF đã được xử lý thành công"}
    except Exception as e:
        logger.error(f"Lỗi khi xử lý PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not vector_store or not rag_agent:
        raise HTTPException(status_code=400, detail="Vui lòng tải PDF trước")

    query = request.query.lower()
    session_id = "user_session_1"

    try:
        # Truy xuất thông tin
        source, context, chunk_index = retriever_agent(query, vector_store, memory, session_id)

        # Tạo prompt và gọi RAG agent
        full_prompt = (
            f"Bối cảnh: {context}\n\n"
            f"Nguồn: {source}\n\n"
            f"Câu hỏi: {query}\n\n"
            "Hãy cung cấp một câu trả lời chi tiết dựa trên thông tin có sẵn."
        )
        logger.debug(f"Prompt gửi đến LLM:\n{full_prompt}")
        response = rag_agent.run(full_prompt)
        answer = response.content

        # Lưu vào lịch sử
        memory.add_to_history(query, answer, session_id=session_id, chunk_index=chunk_index)

        return {"answer": answer}
    except Exception as e:
        logger.error(f"Lỗi khi xử lý câu hỏi: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=List[HistoryItem])
async def get_history(page: int = 1, per_page: int = 25):
    session_id = "user_session_1"
    offset = (page - 1) * per_page
    try:
        with sqlite3.connect(memory.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT query, response, timestamp FROM history
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
            """, (session_id, per_page, offset))
            rows = cursor.fetchall()
            return [{"query": row[0], "response": row[1], "timestamp": row[2]} for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Lỗi khi lấy lịch sử: {e}")
        raise HTTPException(status_code=500, detail=str(e))