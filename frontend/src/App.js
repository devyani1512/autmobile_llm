import React, { useState } from 'react';
import './App.css'; 
import ModeSelector from './components/ModeSelector';
import Chatbot from './components/Chatbot';

function App() {
  
  const [selectedMode, setSelectedMode] = useState(null); 

  return (
    <div className="App">
      <header className="App-header">
        <h1> RAG Manual QA Chatbot </h1>
        <p>Expert Q&A for Technical Manuals</p>
      </header>
      
      <main className="App-main">
        {!selectedMode ? (
          
          <ModeSelector 
            onModeSelect={setSelectedMode} 
          />
        ) : (
          
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