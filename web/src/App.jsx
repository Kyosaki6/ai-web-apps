import { useEffect, useState } from 'react';
import { getHealth } from './api.js';
import Classify from './features/Classify.jsx';
import Detect from './features/Detect.jsx';
import Search from './features/Search.jsx';
import Chat from './features/Chat.jsx';

const TABS = [
  { id: 'classify', label: 'Phân loại ảnh', model: 'classifier', Component: Classify },
  { id: 'detect', label: 'Phát hiện đối tượng', model: 'detector', Component: Detect },
  { id: 'search', label: 'Tìm kiếm ảnh', model: 'retrieval', Component: Search },
  { id: 'chat', label: 'Chatbot RAG', model: 'llm', Component: Chat },
];

export default function App() {
  const [tab, setTab] = useState('classify');
  const [health, setHealth] = useState(null);
  const [showConfig, setShowConfig] = useState(false);
  const [apiUrlInput, setApiUrlInput] = useState(localStorage.getItem('API_URL') || '');

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: 'down', models: {} }));
  }, []);

  const saveApiUrl = (e) => {
    e.preventDefault();
    if (apiUrlInput.trim()) {
      localStorage.setItem('API_URL', apiUrlInput.trim());
    } else {
      localStorage.removeItem('API_URL');
    }
    window.location.reload();
  };

  const current = TABS.find((t) => t.id === tab);
  const ready = health?.models?.[current.model];

  return (
    <div className="app">
      <header>
        <h1>AI Web Apps</h1>
        <p className="muted">
          Backend: {health ? (health.status === 'ok' ? `đang chạy (${health.device})` : 'không kết nối') : 'đang kiểm tra…'}
          <button
            type="button"
            className="button"
            style={{ marginLeft: 12, padding: '2px 10px', fontSize: 13 }}
            onClick={() => setShowConfig(!showConfig)}
          >
            {showConfig ? 'Đóng' : 'Cấu hình API URL'}
          </button>
        </p>
        {showConfig && (
          <form onSubmit={saveApiUrl} className="row" style={{ margin: '8px 0', padding: '10px 14px', background: '#eef3f6', borderRadius: 8 }}>
            <span style={{ fontSize: 13, fontWeight: 500 }}>Backend API URL:</span>
            <input
              type="text"
              value={apiUrlInput}
              onChange={(e) => setApiUrlInput(e.target.value)}
              placeholder="Ví dụ: https://xyz.ngrok-free.app hoặc http://localhost:8000"
              style={{ fontSize: 13 }}
            />
            <button type="submit" className="button" style={{ padding: '6px 14px', fontSize: 13 }}>Lưu & Kết nối</button>
            {localStorage.getItem('API_URL') && (
              <button
                type="button"
                className="button"
                style={{ padding: '6px 14px', fontSize: 13 }}
                onClick={() => {
                  localStorage.removeItem('API_URL');
                  setApiUrlInput('');
                  window.location.reload();
                }}
              >
                Xóa URL
              </button>
            )}
          </form>
        )}
      </header>
      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? 'active' : ''}
                  onClick={() => setTab(t.id)}>
            {t.label}{health && !health.models?.[t.model] ? ' (tắt)' : ''}
          </button>
        ))}
      </nav>
      <main>
        {health && !ready && <p className="error">Mô hình “{current.model}” chưa được nạp ở backend.</p>}
        <current.Component />
      </main>
    </div>
  );
}
