import React from 'react';

function ModeSelector({ onModeSelect }) {
  return (
    <div className="mode-selector-container">
      <h2 className="text-xl font-bold mb-6 text-gray-700">👋 Welcome! Please select your role:</h2>
      <div className="mode-options">
        <button 
          className="mode-button owner-mode"
          onClick={() => onModeSelect('owner')}
        >
          Owner Mode
          <p className="mode-description">
            (Simple, non-technical explanations)
          </p>
        </button>
        <button 
          className="mode-button mechanic-mode"
          onClick={() => onModeSelect('mechanic')}
        >
          🛠️ Mechanic Mode
          <p className="mode-description">
            (Detailed, technical procedures and specs)
          </p>
        </button>
      </div>
    </div>
  );
}

export default ModeSelector;