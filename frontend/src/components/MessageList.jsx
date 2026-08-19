import React, { useEffect, useRef, useState } from 'react';
import { Bot, User, Database, BookOpen, AlertTriangle, ChevronDown, ChevronUp, Sparkles } from 'lucide-react';

export default function MessageList({ messages, isLoading, onSelectSuggestion }) {
  const messagesEndRef = useRef(null);
  const [expandedSources, setExpandedSources] = useState({});

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const toggleSource = (idx) => {
    setExpandedSources((prev) => ({
      ...prev,
      [idx]: !prev[idx],
    }));
  };

  const getActionBadge = (msg) => {
    if (msg.action_taken === 'tool') {
      const isOrder = msg.tool_name === 'check_order_status' || msg.text?.includes('order') || msg.text?.includes('Order');
      return (
        <span className="action-badge tool">
          <Database size={12} />
          {isOrder ? 'Action: Order Lookup' : 'Action: Account Lookup'}
        </span>
      );
    }
    if (msg.action_taken === 'rag') {
      return (
        <span className="action-badge rag">
          <BookOpen size={12} />
          Action: Knowledge Base
        </span>
      );
    }
    if (msg.action_taken === 'escalate') {
      return (
        <span className="action-badge escalate">
          <AlertTriangle size={12} />
          Action: Human Escalation
        </span>
      );
    }
    return null;
  };

  const suggestions = [
    "What is your refund policy?",
    "Where is order 4521?",
    "Is account 1001 active?",
    "How long does shipping take?",
    "I want to speak with a human.",
  ];

  if (messages.length === 0) {
    return (
      <div className="messages-container">
        <div className="empty-state">
          <div className="empty-icon">
            <Sparkles size={28} />
          </div>
          <h3>Welcome to NovaTech Support</h3>
          <p>
            I can check live order & account statuses, answer policy & subscription questions, or connect you with human support.
          </p>
          <div className="suggestion-chips">
            {suggestions.map((s, idx) => (
              <button
                key={idx}
                className="chip-btn"
                onClick={() => onSelectSuggestion(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="messages-container">
      {messages.map((msg, idx) => (
        <div key={idx} className={`message-row ${msg.sender}`}>
          <div className={`avatar ${msg.sender === 'user' ? 'user-avatar' : 'bot-avatar'}`}>
            {msg.sender === 'user' ? <User size={16} /> : <Bot size={16} />}
          </div>

          <div className="message-content">
            <div className="bubble">{msg.text}</div>

            {msg.sender === 'assistant' && (
              <>
                <div className="action-badge-container">
                  {getActionBadge(msg)}
                </div>

                {msg.escalation_id && (
                  <div className="escalation-card">
                    <div>
                      <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>Flagged for Review</div>
                      <span>{msg.escalation_id}</span>
                    </div>
                    <AlertTriangle size={18} color="#fbbf24" />
                  </div>
                )}

                {msg.sources && msg.sources.length > 0 && (
                  <div>
                    <div
                      className="sources-toggle"
                      onClick={() => toggleSource(idx)}
                    >
                      <span>Sources ({msg.sources.length})</span>
                      {expandedSources[idx] ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    </div>
                    {expandedSources[idx] && (
                      <div style={{
                        marginTop: '4px',
                        padding: '6px 10px',
                        background: 'rgba(0, 0, 0, 0.25)',
                        borderRadius: '6px',
                        fontSize: '0.72rem',
                        color: '#9ca3af',
                        fontFamily: 'var(--font-mono)'
                      }}>
                        {msg.sources.map((src, sIdx) => (
                          <div key={sIdx}>• {src}</div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      ))}

      {isLoading && (
        <div className="message-row assistant">
          <div className="avatar bot-avatar">
            <Bot size={16} />
          </div>
          <div className="message-content">
            <div className="bubble" style={{ padding: '8px 14px' }}>
              <div className="typing-dots">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
}
