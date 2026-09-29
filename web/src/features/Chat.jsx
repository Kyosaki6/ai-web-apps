import { useEffect, useRef, useState } from 'react';
import { getChatModels, streamChat } from '../api.js';

const QUICK_PROMPTS = [
  'Đổi trả hàng trong mấy ngày?',
  'Đơn hàng bao nhiêu tiền được miễn phí vận chuyển?',
  'Hạn mức tối đa khi thanh toán COD là bao nhiêu?',
  'Thiết bị điện tử bảo hành bao lâu?',
];

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [showConfig, setShowConfig] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelError, setModelError] = useState('');
  const abortRef = useRef(null);

  // Tự động tải danh sách models khi khởi tạo
  useEffect(() => {
    fetchModelList();
  }, []);

  async function fetchModelList(url = baseUrl, key = apiKey) {
    setLoadingModels(true);
    setModelError('');
    try {
      const data = await getChatModels(url, key);
      const list = data.models || [];
      setModels(list);
      if (list.length > 0 && (!selectedModel || !list.includes(selectedModel))) {
        setSelectedModel(list[0]);
      }
    } catch (err) {
      setModelError(`Không tải được danh sách model: ${err.message}`);
    } finally {
      setLoadingModels(false);
    }
  }

  async function send(e, customText) {
    if (e) e.preventDefault();
    const text = (customText ?? input).trim();
    if (!text || busy) return;

    const history = messages.map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: 'user', content: text }, { role: 'assistant', content: '', sources: [] }]);
    setInput('');
    setBusy(true);
    abortRef.current = new AbortController();

    const patchLast = (fn) => setMessages((m) => [...m.slice(0, -1), fn(m[m.length - 1])]);

    try {
      await streamChat({
        message: text,
        history,
        model: selectedModel,
        baseUrl: baseUrl.trim() || undefined,
        apiKey: apiKey.trim() || undefined,
        signal: abortRef.current.signal,
        onSources: (items) => patchLast((last) => ({ ...last, sources: items })),
        onToken: (t) => patchLast((last) => ({ ...last, content: last.content + t })),
      });
    } catch (err) {
      if (err.name !== 'AbortError') {
        patchLast((last) => ({ ...last, content: `Lỗi: ${err.message}` }));
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="chat">
      <div className="chat-header">
        <div>
          <h2>Trợ lý ShopLite (RAG)</h2>
          <p className="muted">
            Hỏi đáp chính sách cửa hàng ShopLite với tài liệu trích dẫn & kiểm soát an toàn.
          </p>
        </div>
        <button
          type="button"
          className="button config-btn"
          onClick={() => setShowConfig(!showConfig)}
          title="Tùy chỉnh model & OpenAI-compatible endpoint"
        >
          ⚙ {showConfig ? 'Đóng cấu hình' : 'Chọn Model & Endpoint'}
        </button>
      </div>

      {showConfig && (
        <div className="config-panel">
          <div className="config-row">
            <label>
              <b>Model:</b>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={loadingModels || busy}
              >
                {models.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="button small"
              onClick={() => fetchModelList(baseUrl, apiKey)}
              disabled={loadingModels}
            >
              {loadingModels ? 'Đang tải…' : '🔄 Làm mới'}
            </button>
          </div>

          <div className="config-row">
            <label>
              <b>Base URL (OpenAI-compatible):</b>
              <input
                type="text"
                placeholder="https://api.openai.com/v1 hoặc để trống dùng mặc định"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                disabled={busy}
              />
            </label>
          </div>

          <div className="config-row">
            <label>
              <b>API Key:</b>
              <input
                type="password"
                placeholder="sk-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                disabled={busy}
              />
            </label>
            <button
              type="button"
              className="button small"
              onClick={() => fetchModelList(baseUrl, apiKey)}
              disabled={loadingModels}
            >
              Lấy danh sách từ endpoint
            </button>
          </div>

          {modelError && <p className="error small">{modelError}</p>}
        </div>
      )}

      {/* Model indicator bar */}
      <div className="model-bar">
        <span className="badge">Mô hình: {selectedModel || 'Đang tải...'}</span>
        {baseUrl && <span className="badge endpoint">Endpoint: {baseUrl}</span>}
        {messages.length > 0 && (
          <button
            type="button"
            className="clear-btn"
            onClick={() => setMessages([])}
            disabled={busy}
          >
            Xóa hội thoại
          </button>
        )}
      </div>

      {/* Quick Prompts */}
      {messages.length === 0 && (
        <div className="quick-prompts">
          <span className="muted small">Gợi ý nhanh:</span>
          <div className="prompt-chips">
            {QUICK_PROMPTS.map((qp, idx) => (
              <button
                key={idx}
                type="button"
                className="chip"
                onClick={() => send(null, qp)}
                disabled={busy}
              >
                {qp}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="messages" aria-live="polite">
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <p>{m.content || (busy && i === messages.length - 1 ? '…' : '')}</p>
            {m.sources?.length > 0 && (
              <details className="sources-box">
                <summary>Nguồn tài liệu ({m.sources.length})</summary>
                {m.sources.map((s, j) => (
                  <div key={j} className="source-item">
                    <span className="source-name">📄 {s.source}</span>
                    <span className="source-score">Độ tương đồng: {(s.score * 100).toFixed(1)}%</span>
                    <p className="source-snippet">{s.text.slice(0, 180)}…</p>
                  </div>
                ))}
              </details>
            )}
          </div>
        ))}
      </div>

      <form className="row chat-input-bar" onSubmit={(e) => send(e)}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Nhập câu hỏi về chính sách ShopLite..."
          aria-label="Câu hỏi"
          disabled={busy}
        />
        {busy ? (
          <button type="button" className="button stop-btn" onClick={() => abortRef.current?.abort()}>
            Dừng
          </button>
        ) : (
          <button type="submit" className="button" disabled={!input.trim()}>
            Gửi
          </button>
        )}
      </form>
    </section>
  );
}
