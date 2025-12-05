import sys
import os
from pathlib import Path

# Thêm thư mục gốc (D:\LLM\LLM Learning\) vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import  VietnameseEmbedder, FAISSVectorStore, get_rag_agent
from utils import process_pdf
from persistent_memory import PersistentMemory
import shutil
from typing import List
import logging
import sqlite3
import json
import re
from mcp_client.client import MCPClient  # Import MCP Client
from agents import get_mcp_planner_agent, AnswerGeneratorAgent, get_rag_agent

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

# Đường dẫn lưu PDF (tuyệt đối, tránh lệch CWD)
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
PDF_PATH = PDF_DIR / "uploaded.pdf"
os.makedirs(PDF_DIR, exist_ok=True)

# Khởi tạo global variables
memory = PersistentMemory(db_path=str(BASE_DIR / "data" / "memory.db"), max_history=25)
embedder = None
vector_store = None
rag_agent = None
# Đọc MCP endpoint từ env (MCP_SERVER_URL) hoặc mặc định http://localhost:8000
mcp_client = MCPClient()
# Answer generator (Gemini)
answer_agent = AnswerGeneratorAgent(get_rag_agent())

class QueryRequest(BaseModel):
    query: str
    allow_web_search: bool = False

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
        documents = process_pdf(str(PDF_PATH))
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
    query = request.query
    session_id = "user_session_1"

    try:
        # Planner agent: tool-calling, trả JSON
        planner_agent = get_mcp_planner_agent(allow_web_search=request.allow_web_search)
        planner_output = planner_agent.run(f"[SESSION:{session_id}] {query}").content
        try:
            # cố gắng parse JSON thô; nếu có code block, trích phần {...}
            match = re.search(r"{.*}", planner_output, re.DOTALL)
            payload = match.group(0) if match else planner_output
            obj = json.loads(payload)
            source = obj.get("source", "")
            context = obj.get("context", "")
            memory_context = obj.get("memory", "")
        except Exception:
            raise HTTPException(status_code=500, detail="Planner trả về không phải JSON hợp lệ")
        if source == "error":
            raise HTTPException(status_code=500, detail=context or "Planner lỗi")

        # Answer: dùng Gemini sinh câu trả lời từ context + history
        answer = answer_agent.run(query, context, source, memory_context)
        if answer.startswith("Lỗi"):
            raise HTTPException(status_code=500, detail=answer)

        # Lưu lịch sử qua MCP
        try:
            mcp_client.invoke(
                "memory_add",
                {
                    "session_id": session_id,
                    "query": query,
                    "answer": answer,
                    "chunk_index": None,
                },
            )
        except Exception as e:
            logger.warning(f"Lưu lịch sử lỗi (bỏ qua): {e}")

        return {"answer": answer}
    except Exception as e:
        logger.error(f"Lỗi khi xử lý câu hỏi: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history", response_model=List[HistoryItem])
async def get_history(page: int = 1, per_page: int = 25):
    session_id = "user_session_1"
    try:
        # Lấy lịch sử qua MCP Client
        history_lines = mcp_client.invoke(
            "memory_get", {"session_id": session_id, "max_rows": per_page}
        )
        history_items = []
        for line in history_lines:
            # Parse history_lines thành định dạng HistoryItem
            # Giả sử mỗi line có dạng "[timestamp] Query: ... Response: ..."
            try:
                timestamp_end = line.find("] Query: ")
                if timestamp_end == -1:
                    continue
                timestamp = line[1:timestamp_end]
                query_start = timestamp_end + len("] Query: ")
                query_end = line.find("\nResponse: ")
                if query_end == -1:
                    continue
                query = line[query_start:query_end]
                response = line[query_end + len("\nResponse: "):]
                history_items.append(HistoryItem(query=query, response=response, timestamp=timestamp))
            except Exception as e:
                logger.warning(f"Lỗi khi parse lịch sử: {line}, lỗi: {e}")
                continue
        return history_items
    except Exception as e:
        logger.error(f"Lỗi khi lấy lịch sử: {e}")
        raise HTTPException(status_code=500, detail=str(e))
