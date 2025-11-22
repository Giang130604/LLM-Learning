import sys
import os

# Thêm thư mục cha (src/) vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rag_planner import RAGPlanner
if __name__ == "__main__":
    planner = RAGPlanner("http://localhost:8080")
    while True:
        q = input("\nHỏi (thoát=enter trống): ").strip()
        if not q:
            break
        ans = planner.run(q, session_id="demo_session")
        print("\n--- Trả lời ---\n", ans)