import React, { useState } from 'react';
import './App.css'; 
import ModeSelector from './components/ModeSelector';
import Chatbot from './components/Chatbot';

function App() {
  // State to track the currently selected user mode ('owner', 'mechanic', or null)
  const [selectedMode, setSelectedMode] = useState(null); 

  return (
    <div className="App">
      <header className="App-header">
        <h1>📘 RAG Manual QA Chatbot 🤖</h1>
        <p>Expert Q&A for Technical Manuals</p>
      </header>
      
      <main className="App-main">
        {!selectedMode ? (
          // --- Show Mode Selection ---
          <ModeSelector 
            onModeSelect={setSelectedMode} 
          />
        ) : (
          // --- Show Chatbot Interface ---
          <Chatbot 
            mode={selectedMode} 
            onBack={() => setSelectedMode(null)} 
          />
        )}
      </main>
    </div>
  );
}

export default App;