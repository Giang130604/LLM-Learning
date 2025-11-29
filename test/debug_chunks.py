#!/usr/bin/env python3
"""
Debug script để check nội dung các chunk
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from src.utils import process_pdf

# Thiết lập logging
logging.basicConfig(level=logging.WARNING)  

PDF_PATH = "../data/pdfs/SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf"

def debug_chunks():
    """Debug nội dung các chunk"""
    
    print("=== DEBUG CHUNKS CONTENT ===")
    
    try:
        documents = process_pdf(PDF_PATH)
        print(f"Tổng số chunks: {len(documents)}")
        print()
    except Exception as e:
        print(f"Lỗi: {e}")
        return

    for i, doc in enumerate(documents):
        chunk_idx = doc.metadata.get('index', i+1)
        content = doc.page_content
        
        print(f"=== CHUNK {chunk_idx} ===")
        print(f"Length: {len(content)} characters")
        print(f"Content:")
        print(content)
        print(f"{'='*60}")
        print()

if __name__ == "__main__":
    debug_chunks()