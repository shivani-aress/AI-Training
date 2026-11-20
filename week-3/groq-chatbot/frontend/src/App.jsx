
import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:5000";

function App() {
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([]);
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState("");

  const bottomRef = useRef(null);

  // Automatically scroll to the latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [chat, loading]);

  // ==========================================
  // SEND MESSAGE
  // ==========================================

  async function handleSubmit(event) {
    event.preventDefault();

    const userMessage = message.trim();

    if (!userMessage || loading || clearing) {
      return;
    }

    setError("");

    // Display user message immediately
    setChat((previousChat) => [
      ...previousChat,
      {
        role: "user",
        content: userMessage,
      },
    ]);

    setMessage("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          message: userMessage,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || "Something went wrong."
        );
      }

      // Display Groq response
      setChat((previousChat) => [
        ...previousChat,
        {
          role: "assistant",
          content: data.response,
        },
      ]);

    } catch (requestError) {
      setError(requestError.message);

    } finally {
      setLoading(false);
    }
  }


  // ==========================================
  // CLEAR CHAT
  // ==========================================

  async function handleClearChat() {
    if (loading || clearing) {
      return;
    }

    setError("");
    setClearing(true);

    try {
      const response = await fetch(`${API_BASE_URL}/clear`, {
        method: "POST",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || "Could not clear conversation."
        );
      }

      // Clear messages from frontend
      setChat([]);
      setMessage("");

    } catch (clearError) {
      setError(clearError.message);

    } finally {
      setClearing(false);
    }
  }


  // ==========================================
  // CHATBOT INTERFACE
  // ==========================================

  return (
    <div className="app">

      {/* Header */}
      <header className="app-header">

        <div>
          <h1>Groq Chatbot</h1>
          <p className="header-subtitle">
            A simple AI assistant
          </p>
        </div>

        <button
          className="clear-button"
          type="button"
          onClick={handleClearChat}
          disabled={loading || clearing}
        >
          {clearing ? "Clearing..." : "Clear Chat"}
        </button>

      </header>


      {/* Chat Messages */}
      <main className="chat-area">

        {chat.length === 0 ? (

          <div className="welcome-message">
            <h2>Hi! How can I help you today?</h2>
            <p>Send a message to start a conversation.</p>
          </div>

        ) : (

          chat.map((item, index) => (

            <div
              key={index}
              className={`message ${item.role}`}
            >

              <span className="message-role">
                {item.role === "user" ? "You" : "Groq"}
              </span>

              <p>{item.content}</p>

            </div>

          ))

        )}


        {/* Loading Indicator */}
        {loading && (
          <div className="message assistant">
            <span className="message-role">Groq</span>
            <p className="typing">Thinking...</p>
          </div>
        )}

        <div ref={bottomRef} />

      </main>


      {/* Error Message */}
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}


      {/* Message Input */}
      <form
        className="input-area"
        onSubmit={handleSubmit}
      >

        <input
          type="text"
          placeholder="Type your message..."
          value={message}
          onChange={(event) =>
            setMessage(event.target.value)
          }
          disabled={loading || clearing}
        />

        <button
          type="submit"
          disabled={!message.trim() || loading || clearing}
        >
          {loading ? "Sending..." : "Send"}
        </button>

      </form>

    </div>
  );
}

export default App;
