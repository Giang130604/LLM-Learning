import sys
import os

# Thêm thư mục gốc (D:\LLM\LLM Learning\) vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import google.generativeai as genai
import logging
from utils import  VietnameseEmbedder, FAISSVectorStore, process_pdf
from agents import RetrieverAgent, WebSearcherAgent, MemoryManagerAgent, AnswerGeneratorAgent, CoordinatorAgent, get_rag_agent, get_ollama_agent
from persistent_memory import PersistentMemory

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình API
GEMINI_API_KEY = "AIzaSyCpH0z5chYiTwNt0OITLmuifbPhbDugjQU"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

PDF_PATH = "../data/pdfs/SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf"

def main():
    logger.info("Bắt đầu chương trình...")
    print("Đang xử lý file PDF...")
    try:
        documents = process_pdf(PDF_PATH)
        print(f"Đã tạo {len(documents)} chunk từ file PDF.")
    except Exception as e:
        logger.error(f"Lỗi khi xử lý PDF: {e}")
        print(f"Lỗi khi xử lý PDF: {e}")
        return

    # Khởi tạo các thành phần
    try:
        embedder = VietnameseEmbedder()
        vector_store = FAISSVectorStore(documents, embedder)
        gemini_agent = get_rag_agent()  
        ollama_agent = get_ollama_agent(model_name="llama3") 
        memory = PersistentMemory(db_path="../data/memory.db", max_history=10, embedder=embedder)
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo thành phần: {e}")
        print(f"Lỗi khi khởi tạo thành phần: {e}")
        return

    # Khởi tạo các agent
    try:
        retriever_agent = RetrieverAgent(vector_store, ollama_agent)  
        web_searcher_agent = WebSearcherAgent(ollama_agent)  
        memory_manager_agent = MemoryManagerAgent(memory, ollama_agent) 
        answer_generator_agent = AnswerGeneratorAgent(gemini_agent)  
        coordinator_agent = CoordinatorAgent(
            retriever_agent,
            web_searcher_agent,
            memory_manager_agent,
            answer_generator_agent,
            memory
        )
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo agent: {e}")
        print(f"Lỗi khi khởi tạo agent: {e}")
        return

    session_id = "user_session_1"

    while True:
        print("\nNhập câu hỏi của bạn (nhập 'thoát' để kết thúc): ")
        query = input().strip().lower()
        if not query or query == "thoát":
            logger.info("Người dùng đã chọn thoát chương trình.")
            print("Đã thoát chương trình.")
            break

        try:
            print("\nĐang xử lý câu hỏi...")
            answer = coordinator_agent.run(query, session_id)
            print("\n=== Câu trả lời ===")
            print(answer)
        except Exception as e:
            logger.error(f"Lỗi khi xử lý câu hỏi: {e}")
            print(f"Lỗi khi xử lý câu hỏi: {e}")

if __name__ == "__main__":
    main()