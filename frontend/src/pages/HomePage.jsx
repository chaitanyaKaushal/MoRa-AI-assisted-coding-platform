import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, Code2, CheckCircle2, Circle } from 'lucide-react';

export default function HomePage() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  // Track the current step index (1, 2, or 3)
  const [currentStep, setCurrentStep] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  
  const navigate = useNavigate();

  const handleGenerate = async () => {
    setLoading(true);
    setCurrentStep(1);
    setStatusMessage("Initializing...");

    try {
      const response = await fetch('http://localhost:8000/api/generate-problem', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vague_problem: input })
      });

      if (!response.ok) throw new Error(`Server Error: ${response.statusText}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim() !== '');

        for (const line of lines) {
          try {
            const data = JSON.parse(line);

            if (data.error) {
              throw new Error(data.error);
            }

            if (data.step) {
              setCurrentStep(data.step);
              setStatusMessage(data.message);
            }

            if (data.final_result) {
              navigate(`/problem/${data.final_result.problem_id}`, { 
                state: { initialData: data.final_result } 
              });
              return; // Stop processing once we have the result
            }
          } catch (err) {
            // Re-throw if it's the error from data.error, otherwise log parse error
            if (err.message.startsWith("400") || err.message.startsWith("Server")) throw err;
            console.error("JSON Parse Error:", err);
          }
        }
      }

    } catch (e) {
      console.error(e);
      alert(`Error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Helper component for the Status Steps
  const StepIndicator = ({ stepNumber, label }) => {
    const isActive = currentStep === stepNumber;
    const isCompleted = currentStep > stepNumber;

    return (
      <div className={`flex items-center space-x-3 ${isActive || isCompleted ? 'text-blue-400' : 'text-gray-600'}`}>
        {isCompleted ? (
          <CheckCircle2 className="w-5 h-5 text-green-500" />
        ) : isActive ? (
          <Loader2 className="w-5 h-5 animate-spin" />
        ) : (
          <Circle className="w-5 h-5" />
        )}
        <span className={`text-sm font-medium ${isActive ? 'text-white' : ''}`}>{label}</span>
      </div>
    );
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen p-4 bg-gray-900">
      <div className="mb-8 flex flex-col items-center">
        <div className="p-3 bg-blue-600/20 rounded-full mb-4">
          <Code2 className="w-12 h-12 text-blue-400" />
        </div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          MoRa: AI Coding Platform
        </h1>
        <p className="text-gray-400 mt-2">Powered by Fine-Tuned LLMs</p>
      </div>

      <div className="w-full max-w-2xl bg-gray-800 p-6 rounded-lg shadow-xl border border-gray-700 transition-all">
        {/* If loading, show the Stepper, otherwise show the Input */}
        {loading ? (
          <div className="flex flex-col items-center justify-center py-10 space-y-6">
            <h3 className="text-xl text-white font-semibold animate-pulse">
               Building your environment...
            </h3>
            
            <div className="w-full max-w-sm space-y-4 bg-gray-900 p-6 rounded-lg border border-gray-700">
              <StepIndicator stepNumber={1} label="Formalizing Problem Specification" />
              <StepIndicator stepNumber={2} label="Generating Test Suite (The Oracle)" />
              <StepIndicator stepNumber={3} label="Preparing Reference Solution" />
            </div>

            <p className="text-xs text-gray-500 font-mono mt-4">
              Current Status: {statusMessage}
            </p>
          </div>
        ) : (
          <>
            <label className="block mb-2 text-sm font-medium text-gray-300">
              Describe your coding problem
            </label>
            <textarea
              className="w-full h-48 p-4 bg-gray-900 border border-gray-700 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:outline-none font-mono text-sm"
              placeholder="e.g., Given a list of integers, find the two numbers that add up to a specific target. Constraints: N <= 10^5."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button
              onClick={handleGenerate}
              disabled={!input}
              className="w-full mt-4 flex items-center justify-center py-3 bg-blue-600 hover:bg-blue-700 rounded-md font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-blue-500/20 text-white"
            >
              Generate Environment
            </button>
          </>
        )}
      </div>
    </div>
  );
}