import sqlite3
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class PersistentMemory:
    def __init__(self, db_path: str = "../data/memory.db", max_history: int = 10, embedder=None):
        self.db_path = db_path
        self.max_history = max_history
        self.embedder = embedder  # Kept for compatibility, but not used
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        query TEXT,
                        response TEXT,
                        chunk_index INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                logger.info("Khởi tạo cơ sở dữ liệu lịch sử thành công.")
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi khởi tạo cơ sở dữ liệu: {e}")
            raise

    def add_to_history(self, query: str, response: str, session_id: str = "default", chunk_index: Optional[int] = None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO history (session_id, query, response, chunk_index) VALUES (?, ?, ?, ?)",
                    (session_id, query, response, chunk_index)
                )
                conn.commit()
                cursor.execute("""
                    DELETE FROM history WHERE id NOT IN (
                        SELECT id FROM history ORDER BY timestamp DESC LIMIT ?
                    )
                """, (self.max_history,))
                conn.commit()
                logger.debug(f"Đã thêm vào lịch sử: Query={query}, Session={session_id}")
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi thêm vào lịch sử: {e}")

    def get_context(self, query: str, session_id: str = "default", chunk_index: Optional[int] = None, max_rows: int = 10) -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Retrieve recent history for the session, ignoring chunk_index and query filtering
                cursor.execute("""
                    SELECT query, response, timestamp FROM history
                    WHERE session_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (session_id, max_rows))
                rows = cursor.fetchall()

                context = "\n".join([f"[{row[2]}] Query: {row[0]}\nResponse: {row[1]}" for row in rows])
                if context:
                    logger.info(f"Ngữ cảnh lịch sử được truy xuất:\n{context}")
                return context
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi truy xuất lịch sử: {e}")
            return ""

    def clear_session(self, session_id: str = "default"):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
                conn.commit()
                logger.info(f"Đã xóa lịch sử của phiên {session_id}")
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi xóa lịch sử: {e}")