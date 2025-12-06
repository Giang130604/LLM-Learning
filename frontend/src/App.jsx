import { useEffect, useRef, useState } from "react";
import DOMPurify from "dompurify";
import "./style.css";

const API_BASE = "http://127.0.0.1:9000";
const createSessionId = () =>
  (crypto?.randomUUID ? crypto.randomUUID() : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`);

const readJson = (key, fallback) => {
  if (typeof localStorage === "undefined") return fallback;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    return parsed ?? fallback;
  } catch {
    return fallback;
  }
};

const writeJson = (key, value) => {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore quota/parsing errors */
  }
};

// --- API helpers ---
async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload_pdf`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function askQuestion(query, allowWebSearch, sessionId) {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, allow_web_search: allowWebSearch, session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchHistory(sessionId) {
  const res = await fetch(`${API_BASE}/history?session_id=${encodeURIComponent(sessionId || "")}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export default function App() {
  const initialSessionId = useRef(createSessionId()).current;

  const storedSessions = readJson("sessions", null);
  const storedMessages = readJson("messagesBySession", null);
  const storedCurrentSession = readJson("currentSession", null);

  // UI state
  const [sessions, setSessions] = useState(() =>
    Array.isArray(storedSessions) && storedSessions.length
      ? storedSessions
      : [{ id: initialSessionId, title: "Phiên 1" }]
  );
  const [currentSession, setCurrentSession] = useState(() => {
    if (storedCurrentSession && typeof storedCurrentSession === "string") return storedCurrentSession;
    if (Array.isArray(storedSessions) && storedSessions.length) return storedSessions[0].id;
    return initialSessionId;
  });
  const [messagesBySession, setMessagesBySession] = useState(() => {
    if (storedMessages && typeof storedMessages === "object" && !Array.isArray(storedMessages)) return storedMessages;
    return { [initialSessionId]: [] };
  });
  const [inputStr, setInputStr] = useState("");
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [allowWeb, setAllowWeb] = useState(false);
  const [historyList, setHistoryList] = useState([]);
  const [uploadedFile, setUploadedFile] = useState(null);

  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  const currentMessages = messagesBySession[currentSession] || [];

  const updateMessages = (sessionId, updater) => {
    setMessagesBySession((prev) => {
      const existing = prev[sessionId] || [];
      const next = typeof updater === "function" ? updater(existing) : updater;
      return { ...prev, [sessionId]: next };
    });
  };

  // Keep currentSession valid if sessions change
  useEffect(() => {
    if (!sessions.some((s) => s.id === currentSession)) {
      const fallback = sessions[0]?.id || initialSessionId;
      setCurrentSession(fallback);
    }
  }, [sessions, currentSession, initialSessionId]);

  // Persist to localStorage
  useEffect(() => writeJson("sessions", sessions), [sessions]);
  useEffect(() => writeJson("currentSession", currentSession), [currentSession]);
  useEffect(() => writeJson("messagesBySession", messagesBySession), [messagesBySession]);

  // Load history when switching session
  useEffect(() => {
    fetchHistory(currentSession).then(setHistoryList).catch(console.error);
  }, [currentSession]);

  // Scroll to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentMessages, loading]);

  const handleNewChat = () => {
    const newId = createSessionId();
    const newTitle = `Phiên ${sessions.length + 1}`;
    setSessions((prev) => [...prev, { id: newId, title: newTitle }]);
    setCurrentSession(newId);
    setMessagesBySession((prev) => ({ ...prev, [newId]: [] }));
    setHistoryList([]);
    setUploadedFile(null);
    setInputStr("");
  };

  const handleSwitchSession = (sessionId) => {
    setCurrentSession(sessionId);
    setInputStr("");
    setUploadedFile(null);
    if (!messagesBySession[sessionId]) {
      setMessagesBySession((prev) => ({ ...prev, [sessionId]: [] }));
    }
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const sessionId = currentSession;

    try {
      setUploading(true);
      await uploadPdf(file);
      setUploadedFile(file.name);
      updateMessages(sessionId, (prev) => [...prev, { type: "system", text: `Đã tải lên và xử lý: ${file.name}` }]);
    } catch (err) {
      updateMessages(sessionId, (prev) => [...prev, { type: "system", text: `Lỗi upload: ${err.message}` }]);
    } finally {
      setUploading(false);
      e.target.value = null;
    }
  };

  const handleSendMessage = async () => {
    if (!inputStr.trim()) return;

    const query = inputStr;
    const sessionId = currentSession;
    setInputStr("");

    updateMessages(sessionId, (prev) => [...prev, { type: "user", text: query }]);
    setLoading(true);

    try {
      const { answer } = await askQuestion(query, allowWeb, sessionId);
      updateMessages(sessionId, (prev) => [...prev, { type: "bot", text: answer }]);
      const updatedHist = await fetchHistory(sessionId);
      setHistoryList(updatedHist);
    } catch (err) {
      updateMessages(sessionId, (prev) => [...prev, { type: "bot", text: `Lỗi: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleInput = (e) => {
    const target = e.target;
    target.style.height = "auto";
    target.style.height = `${Math.min(target.scrollHeight, 200)}px`;
    setInputStr(target.value);
  };

  const renderAnswerHtml = (text) => {
    if (!text) return "";
    const normalizeInline = (line) =>
      line
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');

    const normalized = text.replace(/\r\n/g, "\n").replace(/\u00a0/g, " ").trim();
    const bulletFriendly = normalized.replace(/(^|[\n\r]|[:.])\s*([*-])\s+/g, "\n$2 ");
    const lines = bulletFriendly
      .split("\n")
      .map((ln) => ln.trim())
      .filter(Boolean);

    const htmlParts = [];
    let listBuffer = [];

    const flushList = () => {
      if (!listBuffer.length) return;
      htmlParts.push("<ul>");
      listBuffer.forEach((item) => htmlParts.push(`<li>${normalizeInline(item)}</li>`));
      htmlParts.push("</ul>");
      listBuffer = [];
    };

    for (const line of lines) {
      const bullet = line.match(/^[*-]\s+(.*)/);
      if (bullet) {
        listBuffer.push(bullet[1]);
      } else {
        flushList();
        htmlParts.push(`<p>${normalizeInline(line)}</p>`);
      }
    }
    flushList();

    return DOMPurify.sanitize(htmlParts.join(""));
  };

  return (
    <div className="shell">
      {/* --- SIDEBAR --- */}
      <aside className="sidebar">
        <div className="logo">
          <i className="fas fa-atom"></i> RAG COSMIC
        </div>

        <button className="new-chat-btn" onClick={handleNewChat}>
          <i className="fas fa-plus"></i> Cuộc trò chuyện mới
        </button>

        <div className="nav">
          <div className="nav-title">Phiên</div>
          {sessions.map((s) => (
            <div
              key={s.id}
              className="history-item"
              onClick={() => handleSwitchSession(s.id)}
              style={
                s.id === currentSession
                  ? { background: "var(--glass-highlight)", color: "var(--text-primary)", border: "1px solid var(--glass-border)" }
                  : {}
              }
            >
              <i className="far fa-comment-alt"></i>
              <span>{s.title}</span>
            </div>
          ))}

          <div className="nav-title">Lịch sử phiên này</div>
          {historyList
            .slice()
            .reverse()
            .map((h, idx) => (
              <div key={idx} className="history-item">
                <i className="far fa-clock"></i>
                <span>{h.query}</span>
              </div>
            ))}
          {historyList.length === 0 && (
            <div style={{ padding: "0 15px", fontSize: "13px", color: "#64748b" }}>Chưa có lịch sử</div>
          )}
        </div>

        <div className="profile">
          <div className="avatar">U</div>
          <div style={{ fontSize: "14px", fontWeight: "500" }}>User</div>
        </div>
      </aside>

      {/* --- MAIN CHAT --- */}
      <main className="main">
        <div className="chat-scroll-area">
          {currentMessages.length === 0 ? (
            /* HERO / EMPTY STATE */
            <div className="hero-container">
              <div className="hero-icon">
                <i className="fas fa-robot"></i>
              </div>
              <div className="hero-text">
                <h1>Xin chào, tôi có thể giúp gì?</h1>
                <p>Hệ thống RAG hỗ trợ tra cứu tài liệu PDF và tìm kiếm Web thông minh.</p>
              </div>
            </div>
          ) : (
            /* MESSAGE LIST */
            currentMessages.map((msg, idx) => {
              const isUser = msg.type === "user";
              const isSystem = msg.type === "system";
              return (
                <div key={idx} className={`message-wrapper ${isUser ? "user" : ""}`}>
                  {!isUser && (
                    <div className="msg-avatar bot">
                      <i className="fas fa-bolt"></i>
                    </div>
                  )}

                  {msg.type === "bot" ? (
                    <div className="msg-content bot-text" dangerouslySetInnerHTML={{ __html: renderAnswerHtml(msg.text) }} />
                  ) : (
                    <div className={`msg-content ${isUser ? "user-text" : "bot-text"}`}>
                      {isSystem ? (
                        <em style={{ color: "#4ade80" }}>
                          <i className="fas fa-check-circle"></i> {msg.text}
                        </em>
                      ) : (
                        msg.text
                      )}
                    </div>
                  )}

                  {isUser && (
                    <div className="msg-avatar user">
                      <i className="fas fa-user"></i>
                    </div>
                  )}
                </div>
              );
            })
          )}
          {loading && (
            <div className="message-wrapper">
              <div className="msg-avatar bot">
                <i className="fas fa-bolt"></i>
              </div>
              <div className="msg-content bot-text" style={{ color: "#94a3b8" }}>
                <i className="fas fa-circle-notch fa-spin"></i> Đang suy nghĩ...
              </div>
            </div>
          )}
          <div ref={chatEndRef}></div>
        </div>

        {/* --- INPUT FLOATING AREA --- */}
        <div className="input-region">
          <div className="input-container">
            {/* File preview */}
            {uploadedFile && (
              <div className="file-preview">
                <i className="fas fa-file-pdf"></i> {uploadedFile}
                <i className="fas fa-times" style={{ cursor: "pointer", marginLeft: 5 }} onClick={() => setUploadedFile(null)}></i>
              </div>
            )}

            <div className="input-row">
              {/* Upload PDF */}
              <input
                type="file"
                ref={fileInputRef}
                accept="application/pdf"
                style={{ display: "none" }}
                onChange={handleFileSelect}
              />
              <button
                className="icon-btn"
                title="Tải lên PDF"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
              >
                {uploading ? <i className="fas fa-spinner fa-spin"></i> : <i className="fas fa-paperclip"></i>}
              </button>

              <button
                className={`icon-btn ${allowWeb ? "active" : ""}`}
                title={allowWeb ? "Tắt tìm kiếm Web" : "Bật tìm kiếm Web"}
                onClick={() => setAllowWeb(!allowWeb)}
              >
                <i className="fas fa-globe"></i>
              </button>

              <textarea
                rows={1}
                placeholder="Nhập câu hỏi của bạn..."
                value={inputStr}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
              ></textarea>

              <button
                className="icon-btn"
                style={{ color: inputStr ? "#3b82f6" : "inherit" }}
                onClick={handleSendMessage}
                disabled={loading || !inputStr.trim()}
              >
                <i className="fas fa-paper-plane"></i>
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
