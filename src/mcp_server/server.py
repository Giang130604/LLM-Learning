
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from utils import web_search, VietnameseEmbedder, FAISSVectorStore, process_pdf
from persistent_memory import PersistentMemory

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG-Tools MCP Server")

TOOL_REGISTRY = {}

def mcp_tool(name: str):
    def decorator(fn):
        TOOL_REGISTRY[name] = fn
        return fn
    return decorator

@app.get("/mcp/discover")
def discover() -> dict:
    return {"tools": list(TOOL_REGISTRY.keys())}

class InvokeRequest(BaseModel):
    tool: str
    args: dict

@app.post("/mcp/invoke")
def invoke(req: InvokeRequest):
    fn = TOOL_REGISTRY.get(req.tool)
    if not fn:
        logger.error(f"Tool not found: {req.tool}")
        raise HTTPException(404, "Tool not found")
    try:
        result = fn(**req.args)
        logger.info(f"Tool {req.tool} invoked successfully with args: {req.args}")
        return {"result": result}
    except Exception as e:
        logger.error(f"Error invoking tool {req.tool}: {str(e)}")
        raise HTTPException(500, str(e))
# === Tool implementations =======================================

@mcp_tool("web_search_tool")
def web_search_tool(query: str, num_results: int = 10) -> List[str]:
    """Tìm snippet Serper API"""
    try:
        logger.info(f"Performing web search for query: {query}")
        results = web_search(query, num_results)
        return results
    except Exception as e:
        logger.error(f"Error in web_search_tool: {str(e)}")
        raise

BASE_DIR = Path(__file__).resolve().parents[2]
PDF_PATH = BASE_DIR / "data" / "pdfs" / "uploaded.pdf"
MEMORY_DB = BASE_DIR / "data" / "memory.db"

_docs_cache = None
_store_cache = None
_embedder = None  

@mcp_tool("retrieve_chunks")
def retrieve_chunks(question: str, top_k: int = 3) -> List[str]:
    """Truy xuất các đoạn PDF liên quan bằng vector store được cache."""
    global _docs_cache, _store_cache, _embedder
    try:
        if _docs_cache is None:
            logger.info(f"Initializing document cache và vector store từ {PDF_PATH} ...")
            _docs_cache = process_pdf(str(PDF_PATH))
            if not _docs_cache:
                logger.warning("No documents retrieved from PDF.")
                return []
            _embedder = VietnameseEmbedder()
            _store_cache = FAISSVectorStore(_docs_cache, _embedder)
            logger.info("Document cache và vector store đã sẵn sàng.")

        logger.info(f"Retrieving chunks for question: {question}")
        chunks = _store_cache.retrieve(question, top_k=top_k)
        result = [c.page_content for c in chunks]
        logger.info(f"Retrieved {len(result)} chunks.")
        return result
    except MemoryError:
        logger.error("MemoryError khi build hoặc query vector store.")
        return []
    except Exception as e:
        logger.error(f"Error in retrieve_chunks: {str(e)}")
        raise

_memory = PersistentMemory(db_path=str(MEMORY_DB), max_history=25)

@mcp_tool("memory_get")
def memory_get(session_id: str, max_rows: int = 10) -> List[str]:
    """Lấy lịch sử hội thoại"""
    try:
        logger.info(f"Retrieving history for session: {session_id}")
        ctx = _memory.get_context("", session_id=session_id, max_rows=max_rows)
        result = ctx.splitlines()
        logger.info(f"Retrieved {len(result)} history entries.")
        return result
    except Exception as e:
        logger.error(f"Error in memory_get: {str(e)}")
        raise

@mcp_tool("memory_add")
def memory_add(
    session_id: str,
    query: str,
    answer: str,
    chunk_index: int | None = None
):
    """Lưu Q/A vào history"""
    try:
        logger.info(f"Adding to history for session: {session_id}, query: {query}")
        _memory.add_to_history(query, answer, session_id, chunk_index)
        logger.info("History entry added successfully.")
        return "ok"
    except Exception as e:
        logger.error(f"Error in memory_add: {str(e)}")
        raise
