import os
import sys
import numpy as np
import faiss
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
import google.generativeai as genai
from agno.agent import Agent
from agno.models.google import Gemini
from langchain_core.embeddings import Embeddings
from datetime import datetime
from typing import List
import logging
# Thêm import cho Vietnamese Embedding
from sentence_transformers import SentenceTransformer

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Cấu hình API key và tham số ---
GEMINI_API_KEY = "AIzaSyCSkPtzL-dI1fgxjCDDvBYxaDYA8z529uQ"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

PDF_PATH = "../data/pdfs/SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf"
SIMILARITY_THRESHOLD = 0.3
POPPLER_PATH = r"D:\PDFPOP\Release-24.08.0-0\poppler-24.08.0\Library\bin"

# --- Vietnamese Embedder ---
class VietnameseEmbedder(Embeddings):
    def __init__(self, model_name="AITeamVN/Vietnamese_Embedding"):
        logger.info(f"Đang tải mô hình Vietnamese Embedding: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Đã tải mô hình Vietnamese Embedding thành công.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embedding cho danh sách các văn bản.
        """
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Lỗi khi tạo embeddings cho documents: {e}")
            return [[0.0] * 768 for _ in texts]  # Trả về vector mặc định nếu có lỗi

    def embed_query(self, text: str) -> List[float]:
        """
        Tạo embedding cho một câu truy vấn.
        """
        try:
            embedding = self.model.encode([text], show_progress_bar=False)[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding cho query: {e}")
            return [0.0] * 768  # Trả về vector mặc định nếu có lỗi

# --- Xử lý PDF ---
def process_pdf(file_path: str) -> List[Document]:
    if not os.path.exists(file_path):
        logger.error(f"File PDF không tồn tại: {file_path}")
        sys.exit(1)
    logger.info(f"Đang xử lý PDF: {file_path}")

    # Sử dụng partition_pdf để trích xuất nội dung từ PDF
    raw_elements = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        poppler_path=POPPLER_PATH,
        dpi=200
    )

    documents = []
    for i, element in enumerate(raw_elements):
        content = str(element)
        doc = Document(
            page_content=content,
            metadata={
                "index": i + 1,
                "file_name": os.path.basename(file_path),
                "timestamp": datetime.now().isoformat()
            }
        )
        documents.append(doc)
    logger.info(f"Đã trích xuất {len(documents)} chunk từ PDF.")
    return documents

# --- FAISS Vector Store ---
class FAISSVectorStore:
    def __init__(self, documents: List[Document], embedder: Embeddings):
        self.documents = documents
        self.embedder = embedder
        logger.info("Đang tạo embeddings cho các chunk...")
        self.embeddings = self.embedder.embed_documents([doc.page_content for doc in documents])
        self.embeddings_np = np.array(self.embeddings).astype("float32")
        norms = np.linalg.norm(self.embeddings_np, axis=1, keepdims=True)
        self.embeddings_np = self.embeddings_np / norms
        d = self.embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embeddings_np)
        logger.info("Đã tạo FAISS index thành công.")

    def retrieve(self, query: str, top_k=3, threshold=SIMILARITY_THRESHOLD) -> List[Document]:
        logger.info(f"Truy xuất tài liệu cho câu hỏi: {query}")
        q_embedding = self.embedder.embed_query(query)
        q_embedding = np.array(q_embedding, dtype="float32")
        q_norm = np.linalg.norm(q_embedding)
        if q_norm > 0:
            q_embedding = q_embedding / q_norm
        q_embedding = np.expand_dims(q_embedding, axis=0)
        D, I = self.index.search(q_embedding, top_k)
        results = []
        for idx, sim in zip(I[0], D[0]):
            if sim >= threshold:
                results.append(self.documents[idx])
                logger.info(f"Đoạn được truy xuất: chunk {idx + 1}, similarity: {sim}")
        return results

# --- Agent (Generation Component) ---
def get_rag_agent() -> Agent:
    return Agent(
        name="Gemini RAG Agent",
        model=Gemini(id="gemini-2.0-flash-thinking-exp-01-21"),
        instructions=(
            "Bạn là một đại lý thông minh chuyên cung cấp câu trả lời chính xác dựa trên tài liệu.\n"
            "Trích dẫn thông tin chi tiết và chính xác, kèm theo số trang nếu có."
        ),
        show_tool_calls=True,
        markdown=True,
    )

# --- Main Execution ---
def main():
    logger.info("Bắt đầu chương trình...")
    print("Đang xử lý file PDF...")
    documents = process_pdf(PDF_PATH)
    print(f"Đã tạo {len(documents)} chunk từ file PDF.")

    # Tạo embedder và vector store
    embedder = VietnameseEmbedder()
    vector_store = FAISSVectorStore(documents, embedder)

    # Nhận câu hỏi từ người dùng
    query = input("Nhập câu hỏi của bạn: ").strip()
    if not query:
        logger.warning("Không có câu hỏi nào được nhập. Thoát.")
        print("Không có câu hỏi nào được nhập. Thoát.")
        return

    # Truy xuất các chunk có liên quan
    retrieved_docs = vector_store.retrieve(query, top_k=3)
    if not retrieved_docs:
        logger.info("Không tìm thấy thông tin phù hợp.")
        print("Không tìm thấy thông tin phù hợp trong tài liệu.")
        return

    # Xây dựng bối cảnh dựa trên các chunk đã truy xuất
    context = "\n\n".join([f"Chunk {doc.metadata.get('index')}: {doc.page_content}" for doc in retrieved_docs])
    print(f"\nĐã tìm thấy {len(retrieved_docs)} chunk có liên quan.")

    # Tạo prompt và sinh câu trả lời
    try:
        rag_agent = get_rag_agent()
        full_prompt = (
            f"Bối cảnh: {context}\n\n"
            f"Câu hỏi: {query}\n\n"
            "Hãy cung cấp một câu trả lời chi tiết dựa trên thông tin có sẵn."
        )
        print("\nĐang sinh câu trả lời...")
        response = rag_agent.run(full_prompt)
        answer = response.content
        print("\n=== Câu trả lời ===")
        print(answer)
    except Exception as e:
        logger.error(f"Lỗi khi sinh câu trả lời: {e}")
        print(f"Lỗi khi sinh câu trả lời: {e}")

if __name__ == "__main__":
    main()