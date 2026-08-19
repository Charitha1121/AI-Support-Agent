import React, { useEffect, useState } from 'react';
import ChatHeader from './components/ChatHeader';
import MessageList from './components/MessageList';
import MessageInput from './components/MessageInput';
import { checkBackendHealth, sendChatMessage } from './services/api';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(() => 'conv-' + Math.random().toString(36).substring(2, 9));
  const [isLoading, setIsLoading] = useState(false);
  const [isHealthy, setIsHealthy] = useState(true);

  useEffect(() => {
    const verifyHealth = async () => {
      const healthy = await checkBackendHealth();
      setIsHealthy(healthy);
    };
    verifyHealth();
    const interval = setInterval(verifyHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleSendMessage = async (text) => {
    if (!text.trim()) return;

    const userMessage = {
      sender: 'user',
      text: text.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const result = await sendChatMessage(text, conversationId);

      const assistantMessage = {
        sender: 'assistant',
        text: result.response,
        action_taken: result.action_taken,
        tool_name: result.tool_name,
        sources: result.sources,
        escalation_id: result.escalation_id,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      if (result.conversation_id) {
        setConversationId(result.conversation_id);
      }
    } catch (err) {
      const errorMessage = {
        sender: 'assistant',
        text: 'Unable to reach the support agent server. Please make sure the backend is running at http://localhost:8000.',
        action_taken: 'escalate',
        escalation_id: null,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setConversationId('conv-' + Math.random().toString(36).substring(2, 9));
  };

  return (
    <div className="app-container">
      <ChatHeader isHealthy={isHealthy} onNewChat={handleNewChat} />
      <main className="chat-canvas">
        <MessageList
          messages={messages}
          isLoading={isLoading}
          onSelectSuggestion={handleSendMessage}
        />
        <MessageInput
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
        />
      </main>
    </div>
  );
}
