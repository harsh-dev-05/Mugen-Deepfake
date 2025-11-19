import React, { useState } from 'react';
import DeepfakeDetector from './components/DeepfakeDetector';

// This is a placeholder for the actual result type from your analysis
type AnalysisResult = {
  verdict: 'Authentic' | 'Deepfake';
  confidence: number;
};

export default function App() {
  const [lastResult, setLastResult] = useState<AnalysisResult | null>(null);

  const handleAnalysisResult = (result: AnalysisResult) => {
    console.log('Analysis complete:', result);
    setLastResult(result);
  };

  return (
    <main className="bg-navy min-h-screen w-full flex flex-col items-center justify-center p-4 font-sans text-slate">
      <div className="w-full max-w-4xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-lightest-slate">
            Deepfake Detection Pipeline
          </h1>
          <p className="text-light-slate mt-2 max-w-2xl mx-auto">
            Upload a video to begin the analysis process. Our AI pipeline will check for signs of digital manipulation.
          </p>
        </header>

        <DeepfakeDetector 
          onResult={handleAnalysisResult}
          simulate={true}                      // <-- TURN OFF SIMULATION
          apiUrl="http://localhost:8000"        // <-- YOUR FASTAPI URL
          className="w-full"
        />


        {lastResult && (
          <div className="text-center mt-8 text-sm">
            <p>Last result received in App component:</p>
            <p className="font-mono bg-light-navy p-2 rounded mt-1">
              {`{ verdict: '${lastResult.verdict}', confidence: ${lastResult.confidence.toFixed(2)} }`}
            </p>
          </div>
        )}
      </div>
    </main>
  );
}