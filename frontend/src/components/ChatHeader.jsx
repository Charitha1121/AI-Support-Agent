import React from 'react';
import { Bot, ShieldCheck, RefreshCw } from 'lucide-react';

export default function ChatHeader({ isHealthy, onNewChat }) {
  return (
    <header className="chat-header">
      <div className="brand-section">
        <div className="brand-logo">
          <Bot size={22} color="#ffffff" />
        </div>
        <div className="brand-info">
          <h1>NovaTech Support</h1>
          <p>Agentic Support with Tool Calling & RAG</p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <button
          onClick={onNewChat}
          className="chip-btn"
          style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          title="Reset conversation session"
        >
          <RefreshCw size={13} />
          <span>New Chat</span>
        </button>

        <div className="header-status">
          <div
            className="status-dot"
            style={{
              backgroundColor: isHealthy ? '#10b981' : '#f59e0b',
              boxShadow: isHealthy ? '0 0 8px #10b981' : '0 0 8px #f59e0b',
            }}
          />
          <span>{isHealthy ? 'Agent Online' : 'Connecting...'}</span>
        </div>
      </div>
    </header>
  );
}
