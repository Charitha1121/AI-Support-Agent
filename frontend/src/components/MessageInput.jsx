import React, { useState } from 'react';
import { Send } from 'lucide-react';

export default function MessageInput({ onSendMessage, isLoading }) {
  const [inputText, setInputText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;
    onSendMessage(inputText);
    setInputText('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-input-area">
      <form onSubmit={handleSubmit} className="input-wrapper">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question, check an order (#4521), or account (#1001)..."
          className="chat-input"
          disabled={isLoading}
          autoFocus
        />
        <button
          type="submit"
          disabled={!inputText.trim() || isLoading}
          className="send-btn"
          title="Send message"
        >
          <Send size={16} />
        </button>
      </form>
    </div>
  );
}
