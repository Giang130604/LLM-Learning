import sys
import os
from pathlib import Path
import shutil
import logging
import json
import re
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import VietnameseEmbedder, FAISSVectorStore, get_rag_agent  
from agents import get_mcp_planner_agent, AnswerGeneratorAgent  
from utils import process_pdf  
from persistent_memory import PersistentMemory  
from mcp_client.client import MCPClient  

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
PDF_PATH = PDF_DIR / "uploaded.pdf"
os.makedirs(PDF_DIR, exist_ok=True)

# Globals
memory = PersistentMemory(db_path=str(BASE_DIR / "data" / "memory.db"), max_history=25)
embedder = None
vector_store = None
rag_agent = None
mcp_client = MCPClient()
answer_agent = AnswerGeneratorAgent(get_rag_agent())


class QueryRequest(BaseModel):
    query: str
    allow_web_search: bool = False
    session_id: str = "user_session_1"


class HistoryItem(BaseModel):
    query: str
    response: str
    timestamp: str


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Nhận file PDF, xử lý thành chunks, khởi tạo embedder + FAISS cho phiên làm việc.
    """
    global embedder, vector_store, rag_agent
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File phải là PDF")

    try:
        with open(PDF_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        documents = process_pdf(str(PDF_PATH))
        logger.info("Đã xử lý PDF, tạo %s chunks", len(documents))

        embedder = VietnameseEmbedder()
        vector_store = FAISSVectorStore(documents, embedder)
        rag_agent = get_rag_agent()

        return {"message": "PDF đã được xử lý thành công"}
    except Exception as e:
        logger.error("Lỗi khi xử lý PDF: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Nhận câu hỏi, dùng planner (MCP tools) để lấy context, sau đó Gemini trả lời.
    Mọi lỗi planner/thiếu dữ liệu sẽ trả lời thân thiện thay vì 500.
    """
    query = request.query
    session_id = request.session_id or "user_session_1"

    try:
        planner_agent = get_mcp_planner_agent(allow_web_search=request.allow_web_search)
        planner_output = planner_agent.run(f"[SESSION:{session_id}] {query}").content

        try:
            match = re.search(r"{.*}", planner_output, re.DOTALL)
            payload = match.group(0) if match else planner_output
            obj = json.loads(payload)
            source = obj.get("source", "")
            context = obj.get("context", "")
            memory_context = obj.get("memory", "")
        except Exception:
            logger.warning("Planner output không parse được JSON: %s", planner_output)
            friendly = "Không đọc được kế hoạch, bạn có thể hỏi lại hoặc bật tìm kiếm web."
            return {"answer": friendly}

        if source == "error":
            logger.warning("Planner trả về error: %s", context)
            friendly = context or "Không lấy được kế hoạch. Thử lại hoặc bật tìm kiếm web."
            return {"answer": friendly}

        answer = answer_agent.run(query, context, source, memory_context)
        # Cho phép trả câu trả lời dạng 'không tìm thấy' như kết quả hợp lệ

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
            logger.warning("Lưu lịch sử lỗi (bỏ qua): %s", e)

        return {"answer": answer}
    except Exception as e:
        logger.error("Lỗi khi xử lý câu hỏi: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=List[HistoryItem])
async def get_history(session_id: str = "user_session_1", page: int = 1, per_page: int = 25):
    try:
        history_lines = mcp_client.invoke(
            "memory_get", {"session_id": session_id, "max_rows": per_page}
        )
        history_items = []
        for line in history_lines:
            try:
                timestamp_end = line.find("] Query: ")
                if timestamp_end == -1:
                    continue
                timestamp = line[1:timestamp_end]
                query_start = timestamp_end + len("] Query: ")
                query_end = line.find("\nResponse: ")
                if query_end == -1:
                    continue
                query_val = line[query_start:query_end]
                response_val = line[query_end + len("\nResponse: "):]
                history_items.append(HistoryItem(query=query_val, response=response_val, timestamp=timestamp))
            except Exception as e:
                logger.warning("Lỗi khi parse lịch sử: %s (line=%s)", e, line)
                continue
        return history_items
    except Exception as e:
        logger.error("Lỗi khi lấy lịch sử: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
