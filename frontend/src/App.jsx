import { useState, useRef, useEffect } from "react";
import DOMPurify from "dompurify";
import "./style.css";

const API_BASE = "http://127.0.0.1:9000";

// --- API Functions giữ nguyên ---
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

async function askQuestion(query, allowWebSearch) {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, allow_web_search: allowWebSearch }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchHistory() {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export default function App() {
  // State quản lý UI
  const [messages, setMessages] = useState([]); // Danh sách tin nhắn hiển thị
  const [inputStr, setInputStr] = useState("");
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false); // Trạng thái đang trả lời
  const [allowWeb, setAllowWeb] = useState(false);
  const [historyList, setHistoryList] = useState([]); // Sidebar history
  const [uploadedFile, setUploadedFile] = useState(null); // Tên file đã upload

  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  // Load history lúc đầu (tùy chỉnh logic nếu muốn hiển thị ngay)
  useEffect(() => {
    fetchHistory().then(setHistoryList).catch(console.error);
  }, []);

  // Cuộn xuống cuối khi có tin nhắn mới
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Hàm xử lý upload file (ẩn input, kích hoạt bằng icon)
  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setUploading(true);
      // Gửi ngay lập tức hoặc chờ user? Ở đây gửi luôn để index
      await uploadPdf(file);
      setUploadedFile(file.name);
      // Thêm thông báo hệ thống vào luồng chat
      setMessages(prev => [...prev, { type: 'system', text: `Đã tải lên và xử lý: ${file.name}` }]);
    } catch (err) {
      setMessages(prev => [...prev, { type: 'system', text: `Lỗi upload: ${err.message}` }]);
    } finally {
      setUploading(false);
      // Reset input để chọn lại cùng file nếu muốn
      e.target.value = null; 
    }
  };

  const handleSendMessage = async () => {
    if (!inputStr.trim()) return;
    
    const query = inputStr;
    setInputStr(""); // Clear input ngay

    // 1. Hiển thị tin nhắn user
    setMessages(prev => [...prev, { type: 'user', text: query }]);
    setLoading(true);

    try {
      // 2. Gọi API
      const { answer } = await askQuestion(query, allowWeb);
      
      // 3. Hiển thị tin nhắn bot
      setMessages(prev => [...prev, { type: 'bot', text: answer }]);
      
      // 4. Update history sidebar
      const updatedHist = await fetchHistory();
      setHistoryList(updatedHist);

    } catch (err) {
      setMessages(prev => [...prev, { type: 'bot', text: `Lỗi: ${err.message}` }]);
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

  // Tự động resize textarea
  const handleInput = (e) => {
    const target = e.target;
    target.style.height = 'auto';
    target.style.height = `${Math.min(target.scrollHeight, 200)}px`;
    setInputStr(target.value);
  };

  const renderAnswerHtml = (text) => {
    if (!text) return "";
    const normalizeInline = (line) => {
      return line
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(
          /(https?:\/\/[^\s<]+)/g,
          '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
        );
    };

    const normalized = text.replace(/\r\n/g, "\n").replace(/\u00a0/g, " ").trim();
    const bulletFriendly = normalized.replace(/(^|[\n\r]|[:.])\s*([*-])\s+/g, "\n$2 ");
    const lines = bulletFriendly.split("\n").map((ln) => ln.trim()).filter(Boolean);

    let htmlParts = [];
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
        <div className="logo"><i className="fas fa-atom"></i> RAG COSMIC</div>
        
        <button className="new-chat-btn" onClick={() => setMessages([])}>
          <i className="fas fa-plus"></i> Cuộc trò chuyện mới
        </button>

        <div className="nav">
          <div className="nav-title">Gần đây</div>
          {historyList.slice().reverse().map((h, idx) => (
            <div key={idx} className="history-item">
              <i className="far fa-comment-alt"></i>
              <span>{h.query}</span>
            </div>
          ))}
          {historyList.length === 0 && <div style={{padding: '0 15px', fontSize: '13px', color: '#64748b'}}>Chưa có lịch sử</div>}
        </div>

        <div className="profile">
          <div className="avatar">U</div>
          <div style={{fontSize: '14px', fontWeight: '500'}}>User</div>
        </div>
      </aside>

      {/* --- MAIN CHAT --- */}
      <main className="main">
        <div className="chat-scroll-area">
          {messages.length === 0 ? (
            /* HERO / EMPTY STATE */
            <div className="hero-container">
              <div className="hero-icon"><i className="fas fa-robot"></i></div>
              <div className="hero-text">
                <h1>Xin chào, tôi có thể giúp gì?</h1>
                <p>Hệ thống RAG hỗ trợ tra cứu tài liệu PDF và tìm kiếm Web thông minh.</p>
              </div>
            </div>
          ) : (
            /* MESSAGE LIST */
            messages.map((msg, idx) => {
              const isUser = msg.type === 'user';
              const isSystem = msg.type === 'system';
              return (
                <div key={idx} className={`message-wrapper ${isUser ? 'user' : ''}`}>
                  {!isUser && (
                    <div className="msg-avatar bot"><i className="fas fa-bolt"></i></div>
                  )}
                  
                  {msg.type === 'bot' ? (
                    <div
                      className="msg-content bot-text"
                      dangerouslySetInnerHTML={{ __html: renderAnswerHtml(msg.text) }}
                    />
                  ) : (
                    <div className={`msg-content ${isUser ? 'user-text' : 'bot-text'}`}>
                      {isSystem ? (
                        <em style={{color: '#4ade80'}}><i className="fas fa-check-circle"></i> {msg.text}</em>
                      ) : (
                        msg.text
                      )}
                    </div>
                  )}

                  {isUser && (
                    <div className="msg-avatar user"><i className="fas fa-user"></i></div>
                  )}
                </div>
              );
            })
          )}
          {loading && (
             <div className="message-wrapper">
                <div className="msg-avatar bot"><i className="fas fa-bolt"></i></div>
                <div className="msg-content bot-text" style={{color: '#94a3b8'}}>
                  <i className="fas fa-circle-notch fa-spin"></i> Đang suy nghĩ...
                </div>
             </div>
          )}
          <div ref={chatEndRef}></div>
        </div>

        {/* --- INPUT FLOATING AREA --- */}
        <div className="input-region">
          <div className="input-container">
            {/* Hiển thị file đã chọn (nếu có) */}
            {uploadedFile && (
              <div className="file-preview">
                <i className="fas fa-file-pdf"></i> {uploadedFile}
                <i className="fas fa-times" style={{cursor: 'pointer', marginLeft: 5}} onClick={() => setUploadedFile(null)}></i>
              </div>
            )}

            <div className="input-row">
              {/* Nút Upload PDF ẩn */}
              <input 
                type="file" 
                ref={fileInputRef} 
                accept="application/pdf" 
                style={{display: 'none'}} 
                onChange={handleFileSelect} 
              />
              <button 
                className="icon-btn" 
                title="Tải lên PDF" 
                onClick={() => fileInputRef.current.click()}
                disabled={uploading}
              >
                {uploading ? <i className="fas fa-spinner fa-spin"></i> : <i className="fas fa-paperclip"></i>}
              </button>
              
              <button 
                className={`icon-btn ${allowWeb ? 'active' : ''}`} 
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
                style={{color: inputStr ? '#3b82f6' : 'inherit'}}
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

