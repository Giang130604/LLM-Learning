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
import urllib.parse
import requests
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = "AIzaSyCSkPtzL-dI1fgxjCDDvBYxaDYA8z529uQ"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

SERPER_API_KEY = "b91e335ef3ef0b0f01dceef77c1c057d0d538bed"

PDF_PATH = "../data/pdfs/SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf"
SIMILARITY_THRESHOLD = 0.3
POPPLER_PATH = r"D:\PDFPOP\Release-24.08.0-0\poppler-24.08.0\Library\bin"


class VietnameseEmbedder(Embeddings):
    # --- Vietnamese Embedder ---
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
            return [[0.0] * 768 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Tạo embedding cho một câu truy vấn.
        """
        try:
            embedding = self.model.encode([text], show_progress_bar=False)[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding cho query: {e}")
            return [0.0] * 768


def process_pdf(file_path: str) -> List[Document]:
    # --- Xử lý PDF ---
    if not os.path.exists(file_path):
        logger.error(f"File PDF không tồn tại: {file_path}")
        sys.exit(1)
    logger.info(f"Đang xử lý PDF: {file_path}")
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


class FAISSVectorStore:
    # --- FAISS Vector Store ---
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


def web_search(query: str, num_results=3) -> List[str]:
    # --- Web Search Tool (Tích hợp từ arag.py) ---
    """
    Tìm kiếm thông tin trên web bằng Serper API.
    """
    encoded_query = urllib.parse.quote(query)
    url = f"https://google.serper.dev/search?q={encoded_query}&apiKey={SERPER_API_KEY}"
    try:
        response = requests.get(url)
        json_data = response.json()
        results = json_data.get("organic", [])
        snippets = [item.get("snippet", "") for item in results[:num_results]]
        return snippets if snippets else ["(Không có kết quả web)"]
    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm web: {e}")
        return ["(Không có kết quả web)"]


def retriever_agent(query: str, vector_store: FAISSVectorStore) -> tuple[str, str]:
    # --- Retriever Agent (Router) ---
    """
    Định tuyến câu hỏi đến nguồn phù hợp: Web Search hoặc Vector Store.
    """
    query_lower = query.lower()
    retrieved_docs = vector_store.retrieve(query_lower, top_k=3)
    if not retrieved_docs:
        logger.info("Không tìm thấy thông tin trong tài liệu. Định tuyến đến Web Search...")
        return "web_search", "\n".join(web_search(query))
    logger.info("Định tuyến đến Vector Store (PDF Documents)...")
    context = "\n\n".join([f"Chunk {doc.metadata.get('index')}: {doc.page_content}" for doc in retrieved_docs])
    return "vector_store", context


def get_rag_agent() -> Agent:
    # --- Agent (Generation Component) ---
    return Agent(
        name="Gemini RAG Agent",
        model=Gemini(id="gemini-2.0-flash-thinking-exp-01-21"),
        instructions=(
            "Bạn là một đại lý thông minh chuyên cung cấp câu trả lời chính xác dựa trên tài liệu.\n"
            "Nếu thông tin đến từ tài liệu PDF, trích dẫn chi tiết và chính xác, kèm theo số trang nếu có.\n"
            "Nếu thông tin đến từ web, hãy ghi rõ nguồn là 'Web Search' và trả lời tự nhiên dựa trên thông tin đó."
        ),
        show_tool_calls=True,
        markdown=True,
    )


def main():
    # --- Main Execution ---
    logger.info("Bắt đầu chương trình...")
    print("Đang xử lý file PDF...")
    documents = process_pdf(PDF_PATH)
    print(f"Đã tạo {len(documents)} chunk từ file PDF.")
    embedder = VietnameseEmbedder()
    vector_store = FAISSVectorStore(documents, embedder)
    rag_agent = get_rag_agent()
    while True:
        print("\nNhập câu hỏi của bạn (nhập 'thoát' để kết thúc): ")
        query = input().strip().lower()
        if not query or query == "thoát":
            logger.info("Người dùng đã chọn thoát chương trình.")
            print("Đã thoát chương trình.")
            break
        source, context = retriever_agent(query, vector_store)
        print(f"\nNguồn: {source}")
        try:
            full_prompt = (
                f"Bối cảnh: {context}\n\n"
                f"Nguồn: {source}\n\n"
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
