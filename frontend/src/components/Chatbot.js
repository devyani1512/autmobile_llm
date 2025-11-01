import React, { useState } from 'react';

// CRITICAL: The base URL of your running FastAPI server
const API_BASE_URL = 'http://localhost:8000';
const ASK_API_ENDPOINT = `${API_BASE_URL}/api/ask`;

// --- Helper Component to Display API Response with Assets ---
// --- Helper Component to Display API Response with Assets ---
const AnswerDisplay = ({ answer }) => {
  if (!answer) return null;

  // Support both {answer_text, images, tables} object and plain string
  const textAnswer =
    typeof answer === 'string'
      ? answer
      : answer.answer_text || '';

  const images =
    Array.isArray(answer.images) ? answer.images : [];

  const tables =
    Array.isArray(answer.tables) ? answer.tables : [];

  // Regex to find asset paths (for backward compatibility)
  const imageMatches = [...textAnswer.matchAll(/\[IMAGE:\s*([^\]]+)\]/g)];
  const tableMatches = [...textAnswer.matchAll(/\[TABLE:\s*([^\]]+)\]/g)];

  // Merge any explicit assets from backend
  const allImages = [...imageMatches.map(m => m[1]), ...images];
  const allTables = [...tableMatches.map(m => m[1]), ...tables];

  // Clean text for display
  const cleanText = textAnswer
    .replace(/\[IMAGE:\s*([^\]]+)\]/g, '')
    .replace(/\[TABLE:\s*([^\]]+)\]/g, '')
    .trim();

  const getAssetUrl = (path) => {
    const filename = path.split('/').pop().trim();
    return `${API_BASE_URL}/assets/${filename}`;
  };

  return (
    <div className="answer-box">
      <h4 className="answer-title">
        ✅ Answer ({cleanText ? 'Grounded Response' : 'No Information Found'}):
      </h4>

      {cleanText ? (
        <p className="answer-text" style={{ whiteSpace: 'pre-wrap' }}>
          {cleanText}
        </p>
      ) : (
        <p className="answer-text text-gray-500">
          No specific information found for that query in the indexed manual(s).
        </p>
      )}

      {(allImages.length > 0 || allTables.length > 0) && (
        <div className="asset-section">
          <h5 className="asset-heading">🔗 Related Manual Assets:</h5>

          {/* Display Images */}
          {allImages.map((path, index) => {
            const url = getAssetUrl(path);
            const filename = path.split('/').pop();

            return (
              <div key={`img-${index}`} className="asset-item image-item">
                <img
                  src={url}
                  alt={`Manual illustration: ${filename}`}
                  className="manual-image"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.style.display = 'none';
                    console.error(`Failed to load asset: ${url}`);
                  }}
                />
                <p className="asset-caption">
                  Figure {index + 1}: {filename}
                </p>
              </div>
            );
          })}

          {/* Display Tables */}
          {allTables.map((path, index) => {
            const url = getAssetUrl(path);
            const filename = path.split('/').pop();

            return (
              <div key={`tbl-${index}`} className="asset-item table-item">
                <p className="asset-caption">📊 Table {index + 1}:</p>
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="asset-link"
                >
                  Download/View CSV ({filename})
                </a>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};



// --- Main Chatbot Component ---
function Chatbot({ mode, onBack }) {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!query.trim()) {
        setError("Please enter a question.");
        return;
    }

    setLoading(true);
    setAnswer(null);
    setError(null);

    try {
      const response = await fetch(ASK_API_ENDPOINT, { 
        method: 'POST', // CRITICAL: Use POST
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: query, 
          mode: mode 
        }),
      });

      if (!response.ok) {
        // Attempt to read the error detail from the response body
        const errorData = await response.json().catch(() => ({ detail: `HTTP status ${response.status}` }));
        throw new Error(`API Request Failed: ${errorData.detail}`);
      }

      const data = await response.json();
      setAnswer(data.answer); // Assumes API returns { "answer": "..." }

    } catch (err) {
      console.error("API Fetch Error:", err);
      setError(` Failed to get answer: ${err.message}. Please ensure your FastAPI server is running at ${API_BASE_URL}.`);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="chatbot-container">
      <button onClick={onBack} className="back-button">
        ← Change Role ({mode.toUpperCase()})
      </button>
      
      <h2 className="chatbot-heading">Ask the Manual</h2>
      
      <div className="input-group">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={`Enter your technical question for the ${mode} mode...`}
          disabled={loading}
        />
        <button onClick={handleSearch} disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}
      
      {loading && <div className="loading-message">...Analyzing manual content for the best match...</div>}
      
      <AnswerDisplay answer={answer} />
      
      {!loading && !answer && !error && (
        <div className="tip-message">
          Type your query above to start. Remember your selected role dictates the response style.
        </div>
      )}
    </div>
  );
}

export default Chatbot;