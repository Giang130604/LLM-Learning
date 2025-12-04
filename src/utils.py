import os
from pathlib import Path
import numpy as np
import faiss
# from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from datetime import datetime
from typing import List
import logging
import urllib.parse
import requests
from sentence_transformers import SentenceTransformer
import pickle
import hashlib

from unstructured.partition.pdf import partition_pdf

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.3  
# Đường dẫn tuyệt đối tới thư mục cache (tránh lệch CWD giữa app và MCP)
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "cache"

# Tạo thư mục cache
os.makedirs(CACHE_DIR, exist_ok=True)

# Helper: Tính MD5 hash của file
def get_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# Xử lý PDF
def _clean_text(text: str) -> str:
    """Làm sạch nhẹ để giảm nhiễu OCR (bảng, ký tự thừa)."""
    text = text.replace("|", " ")
    text = text.replace("\u00a0", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def process_pdf(file_path: str) -> List[Document]:
    if not os.path.exists(file_path):
        logger.error(f"File PDF không tồn tại: {file_path}")
        raise FileNotFoundError(f"File PDF không tồn tại: {file_path}")

    pdf_name = os.path.basename(file_path)
    cache_file = os.path.join(CACHE_DIR, f"{pdf_name}.pkl")
    cache_metadata_file = os.path.join(CACHE_DIR, f"{pdf_name}_metadata.pkl")

    pdf_hash = get_file_hash(file_path)

    if os.path.exists(cache_file) and os.path.exists(cache_metadata_file):
        try:
            with open(cache_metadata_file, "rb") as f:
                cached_hash = pickle.load(f)
            if cached_hash == pdf_hash:
                logger.info(f"Đang load chunks từ cache: {cache_file}")
                with open(cache_file, "rb") as f:
                    documents = pickle.load(f)
                logger.info(f"Đã load {len(documents)} chunks từ cache.")
                return documents
            else:
                logger.info("File PDF đã thay đổi, xử lý lại...")
        except Exception as e:
            logger.error(f"Lỗi khi load cache: {e}, xử lý lại PDF...")

    logger.info(f"Đang xử lý PDF: {file_path}")

    def _partition(strategy: str, use_ocr: bool):
        return partition_pdf(
            filename=file_path,
            strategy=strategy,
            infer_table_structure=True if use_ocr else False, 
            extract_images_in_pdf=True,
            languages=["vie"],
            ocr_languages=["vie"] if use_ocr else None,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            pdf_image_dpi = 300
        )

    raw_elements = _partition(strategy="hi_res", use_ocr=True)

    documents = []
    for i, element in enumerate(raw_elements):
        content = _clean_text(str(element))
        doc = Document(
            page_content=content,
            metadata={
                "index": i + 1,
                "file_name": os.path.basename(file_path),
                "timestamp": datetime.now().isoformat()
            }
        )
        documents.append(doc)

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(documents, f)
        with open(cache_metadata_file, "wb") as f:
            pickle.dump(pdf_hash, f)
        logger.info(f"Đã lưu {len(documents)} chunks vào cache: {cache_file}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu cache: {e}")

    logger.info(f"Đã trích xuất {len(documents)} chunk từ PDF.")
    return documents

# Embedding
class VietnameseEmbedder(Embeddings):
    def __init__(self, model_name="AITeamVN/Vietnamese_Embedding"):
        logger.info(f"Đang tải mô hình Vietnamese Embedding: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Đã tải mô hình Vietnamese Embedding thành công.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Lỗi khi tạo embeddings cho documents: {e}")
            return [[0.0] * 768 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode([text], show_progress_bar=False)[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding cho query: {e}")
            return [0.0] * 768

# FAISS Vector Store
class FAISSVectorStore:
    def __init__(self, documents: List[Document], embedder: Embeddings):
        self.documents = documents
        self.embedder = embedder
        logger.info(f"[DEBUG] Khởi tạo FAISSVectorStore với {len(documents)} documents")
        logger.info(f"[DEBUG] Similarity threshold mặc định: {SIMILARITY_THRESHOLD}")
        
        logger.info("Đang tạo embeddings cho các chunk...")
        self.embeddings = self.embedder.embed_documents([doc.page_content for doc in documents])
        logger.info(f"[DEBUG] Đã tạo {len(self.embeddings)} embeddings")
        logger.info(f"[DEBUG] Embedding dimension: {len(self.embeddings[0]) if self.embeddings else 'Unknown'}")
        
        self.embeddings_np = np.array(self.embeddings).astype("float32")
        logger.info(f"[DEBUG] Embeddings array shape: {self.embeddings_np.shape}")
        
        norms = np.linalg.norm(self.embeddings_np, axis=1, keepdims=True)
        self.embeddings_np = self.embeddings_np / norms
        logger.info(f"[DEBUG] Embeddings đã được normalize")
        
        d = self.embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embeddings_np)
        logger.info(f"[DEBUG] FAISS index đã được tạo với {self.index.ntotal} vectors")
        logger.info("Đã tạo FAISS index thành công.")

    def retrieve(self, query: str, top_k=5, threshold=SIMILARITY_THRESHOLD) -> List[Document]:
        logger.info(f"Truy xuất tài liệu cho câu hỏi: {query}")
        logger.info(f"[DEBUG] Similarity threshold hiện tại: {threshold}")
        logger.info(f"[DEBUG] Top K: {top_k}")
        logger.info(f"[DEBUG] Tổng số documents trong vector store: {len(self.documents)}")
        
        q_embedding = self.embedder.embed_query(query)
        logger.info(f"[DEBUG] Query embedding shape: {len(q_embedding)}")
        
        q_embedding = np.array(q_embedding, dtype="float32")
        q_norm = np.linalg.norm(q_embedding)
        logger.info(f"[DEBUG] Query embedding norm: {q_norm}")
        
        if q_norm > 0:
            q_embedding = q_embedding / q_norm
        q_embedding = np.expand_dims(q_embedding, axis=0)
        
        D, I = self.index.search(q_embedding, top_k)
        logger.info(f"[DEBUG] Top {top_k} similarity scores: {D[0].tolist()}")
        logger.info(f"[DEBUG] Top {top_k} document indices: {I[0].tolist()}")
        
        results = []
        found_any = False
        for i, (idx, sim) in enumerate(zip(I[0], D[0])):
            logger.info(f"[DEBUG] Chunk {idx + 1}: similarity = {sim:.4f}, threshold = {threshold}")
            if sim >= threshold:
                results.append(self.documents[idx])
                logger.info(f"✓ Đoạn được truy xuất: chunk {idx + 1}, similarity: {sim:.4f}")
                found_any = True
            else:
                logger.info(f"✗ Đoạn bị loại bỏ: chunk {idx + 1}, similarity {sim:.4f} < threshold {threshold}")
        
        if not found_any and threshold > 0.05:
            logger.info(f"[DEBUG] Không tìm thấy kết quả với threshold {threshold}, thử lại với 0.05")
            fallback_threshold = 0.05
            for i, (idx, sim) in enumerate(zip(I[0], D[0])):
                if sim >= fallback_threshold:
                    results.append(self.documents[idx])
                    logger.info(f"✓ [FALLBACK] Đoạn được truy xuất: chunk {idx + 1}, similarity: {sim:.4f}")
        
        logger.info(f"[DEBUG] Số lượng documents được trả về: {len(results)}")
        return results

# Web searching tool
def web_search(query: str, num_results=10, api_key="b91e335ef3ef0b0f01dceef77c1c057d0d538bed") -> List[str]:
    encoded_query = urllib.parse.quote(query)
    url = f"https://google.serper.dev/search?q={encoded_query}&apiKey={api_key}"
    try:
        response = requests.get(url)
        json_data = response.json()
        results = json_data.get("organic", [])
        snippets = [item.get("snippet", "") for item in results[:num_results]]
        return snippets if snippets else ["(Không có kết quả web)"]
    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm web: {e}")
        return ["(Không có kết quả web)"]
