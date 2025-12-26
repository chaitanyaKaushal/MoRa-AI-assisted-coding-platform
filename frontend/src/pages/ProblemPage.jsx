import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useLocation, useParams, useNavigate } from 'react-router-dom';
import Split from 'react-split';
import Editor from '@monaco-editor/react';
import { 
  Play, Lock, Unlock, AlertTriangle, Eye, ArrowLeft, 
  Plus, Trash2, Send, ChevronRight, Terminal, CheckCircle, XCircle, 
  Loader, RefreshCw, FileText, List, Sliders, Tag, Hash 
} from 'lucide-react';

export default function ProblemPage() {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  
  // --- Data State ---
  const [problemData, setProblemData] = useState(null);
  const [userCode, setUserCode] = useState('');
  
  // --- UI State ---
  const [activeTab, setActiveTab] = useState('desc');
  const [isExecuting, setIsExecuting] = useState(false);
  const [submissionResult, setSubmissionResult] = useState(null);

  // --- Analysis/Healing State ---
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisLogs, setAnalysisLogs] = useState([]); 
  const [analysisResult, setAnalysisResult] = useState(null); 
  const logsEndRef = useRef(null);

  // --- Wrong Solution State ---
  const [wrongCases, setWrongCases] = useState([{ input: '', output: '' }]);

  // --- Penalty State ---
  const [hasUnlockedSolution, setHasUnlockedSolution] = useState(false);

  const maxPossibleScore = hasUnlockedSolution ? 0 : 100;

  // ---------------------------------------------------------------------------
  // 1. REFACTORED FETCH LOGIC
  // ---------------------------------------------------------------------------
  const fetchProblemDetails = useCallback(async (silent = false) => {
    try {
      const res = await fetch(`http://localhost:8000/api/problem/${id}`);
      if (!res.ok) throw new Error('Failed to fetch problem');
      const details = await res.json();
      
      setProblemData(details);
      
      // Only set user code on initial load, don't overwrite user's work on silent refresh
      if (!silent) {
        if (location.state?.initialData?.starter_code) {
          setUserCode(location.state.initialData.starter_code);
        } else {
          setUserCode(details.starter_code || '');
        }
      }
      return details;
    } catch (e) {
      console.error("Error fetching problem:", e);
    }
  }, [id, location.state]);

  // Initial Load
  useEffect(() => {
    fetchProblemDetails();
  }, [fetchProblemDetails]);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [analysisLogs]);

  // ---------------------------------------------------------------------------
  // EXECUTION HANDLER
  // ---------------------------------------------------------------------------
  const handleRun = async () => {
    setIsExecuting(true);
    setSubmissionResult(null);
    try {
      const res = await fetch('http://localhost:8000/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          user_code: userCode,
          problem_id: id,
          user_id: "user_frontend_demo" 
        })
      });
      
      const data = await res.json();
      setSubmissionResult(data);
      setActiveTab('results');
    } catch (e) {
      alert("Execution failed");
    } finally {
      setIsExecuting(false);
    }
  };

  // --- Handlers for Wrong Solution Tab ---
  const addCase = () => setWrongCases([...wrongCases, { input: '', output: '' }]);
  
  const removeCase = (index) => {
    if (wrongCases.length > 1) {
      setWrongCases(wrongCases.filter((_, i) => i !== index));
    }
  };

  const updateCase = (index, field, value) => {
    const newCases = [...wrongCases];
    newCases[index][field] = value;
    setWrongCases(newCases);
  };

  // ---------------------------------------------------------------------------
  // 2. UPDATED ANALYZE HANDLER WITH REFRESH LOGIC
  // ---------------------------------------------------------------------------
  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setAnalysisLogs([]);
    setAnalysisResult(null);

    try {
      const res = await fetch(`http://localhost:8000/api/improve/problem/${id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ examples: wrongCases }) 
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim() !== '');

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            
            if (data.step || data.message) {
              setAnalysisLogs(prev => [...prev, data]);
            }
            if (data.error) {
              setAnalysisLogs(prev => [...prev, { error: data.error }]);
            }

            if (data.final_result) {
              setAnalysisResult(data.final_result);

              if (data.final_result.decision === "YES") {
                setAnalysisLogs(prev => [...prev, { message: "🔄 Refreshing User Interface with new data..." }]);
                await fetchProblemDetails(true);
                setAnalysisLogs(prev => [...prev, { message: "✅ UI Updated. Check 'Test Suite' tab." }]);
              }
            }
          } catch (err) {
            console.error("Stream parse error", err);
          }
        }
      }
    } catch (e) {
      setAnalysisLogs(prev => [...prev, { error: "Network connection failed." }]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const unlockSolution = () => {
    if (window.confirm("Revealing the reference solution will reduce your maximum possible score to 0%. Continue?")) {
      setHasUnlockedSolution(true);
    }
  };

  // Helper to check if form is valid
  const isChallengeValid = !wrongCases.some(c => !c.input.trim() || !c.output.trim());

  if (!problemData) return <div className="h-screen flex items-center justify-center bg-gray-900 text-white">Loading Environment...</div>;

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white overflow-hidden">
      {/* Utility Style to hide Scrollbar */}
      <style>{`
        .scrollbar-hide::-webkit-scrollbar {
            display: none;
        }
        .scrollbar-hide {
            -ms-overflow-style: none;
            scrollbar-width: none;
        }
      `}</style>

      {/* Header */}
      <header className="h-14 bg-gray-800 border-b border-gray-700 flex justify-between items-center px-4 flex-shrink-0">
        <div className="flex items-center space-x-4 overflow-hidden">
          <button onClick={() => navigate('/')} className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-full transition-all">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="h-6 w-px bg-gray-600 mx-2"></div>
          <h2 className="font-semibold text-lg truncate">{problemData.title}</h2>
          <div className="flex items-center px-2 py-1 rounded bg-gray-700 text-xs font-mono text-gray-300">
            <span>Max Score: </span>
            <span className={`ml-1 font-bold ${maxPossibleScore === 100 ? 'text-green-400' : 'text-red-400'}`}>
              {maxPossibleScore}
            </span>
          </div>
        </div>
        
        <button
          onClick={handleRun}
          disabled={isExecuting}
          className={`flex items-center px-6 py-1.5 rounded font-medium transition-colors ${
            isExecuting ? 'bg-gray-600 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700 text-white'
          }`}
        >
          {isExecuting ? 'Running...' : <><Play className="w-4 h-4 mr-2" /> Submit</>}
        </button>
      </header>

      <Split className="flex-1 flex overflow-hidden" sizes={[45, 55]} minSize={300} gutterSize={10}>
        {/* LEFT PANE */}
        <div className="flex flex-col h-full bg-gray-900 border-r border-gray-700">
          
          {/* --- TABS SECTION --- */}
          <div className="flex border-b border-gray-700 bg-gray-800/50 overflow-x-auto scrollbar-hide">
            {[
              { id: 'desc', label: 'Description', icon: FileText },
              { id: 'tests', label: 'Test Suite', icon: Terminal },
              { id: 'solution', label: 'Reference', icon: hasUnlockedSolution ? null : Lock },
              { id: 'wrong-ref', label: 'Challenge AI', icon: AlertTriangle },
              ...(submissionResult ? [{ id: 'results', label: 'Results' }] : [])
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center px-4 py-3 text-sm font-medium transition-colors border-b-2 whitespace-nowrap flex-shrink-0 ${
                  activeTab === tab.id 
                    ? 'border-blue-500 text-blue-400 bg-gray-800' 
                    : 'border-transparent text-gray-400 hover:text-gray-200'
                }`}
              >
                {tab.icon && <tab.icon className="w-3 h-3 mr-2" />}
                {tab.label}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto p-6 relative">
            {/* 1. DESCRIPTION */}
            {activeTab === 'desc' && (
              <div className="prose prose-invert max-w-none">
                <div className="mb-6">
                  <h3 className="text-xl font-bold mb-3 text-gray-100 flex items-center">
                    <FileText className="w-5 h-5 mr-2 text-blue-400" />
                    Problem Description
                  </h3>
                  <p className="text-gray-300 whitespace-pre-wrap ml-1">{problemData.description}</p>
                </div>

                <div className="mb-6">
                  <h4 className="font-bold text-gray-200 mb-2 flex items-center">
                    <List className="w-4 h-4 mr-2 text-yellow-400" />
                    Examples
                  </h4>
                  <div className="bg-gray-800 rounded-lg p-4 font-mono text-sm text-gray-300 whitespace-pre-wrap border border-gray-700 relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-1 h-full bg-yellow-400/50"></div>
                    {problemData.examples}
                  </div>
                </div>

                <div className="mb-6">
                  <h4 className="font-bold text-gray-200 mb-2 flex items-center">
                    <Sliders className="w-4 h-4 mr-2 text-red-400" />
                    Constraints
                  </h4>
                  <code className="flex items-center bg-gray-800 p-3 rounded text-sm text-blue-300 border border-gray-700">
                    <AlertTriangle className="w-3 h-3 mr-2 text-red-400 flex-shrink-0" />
                    <span className="whitespace-pre-wrap">{problemData.constraints}</span>
                  </code>
                </div>

                {/* TAGS SECTION */}
                {problemData.tags && problemData.tags.length > 0 && (
                  <div className="mt-10 pt-6 border-t border-gray-700">
                    <div className="flex items-center text-gray-400 text-sm mb-3">
                      <Tag className="w-4 h-4 mr-2" />
                      <span className="font-semibold uppercase tracking-wider text-xs">Related Topics</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {problemData.tags.map((tag, index) => (
                        <span 
                          key={index} 
                          className="px-3 py-1 bg-gray-800 text-gray-300 text-xs font-medium rounded-full border border-gray-600 hover:bg-gray-700 hover:border-gray-500 hover:text-white transition-colors cursor-default flex items-center"
                        >
                          <Hash className="w-3 h-3 mr-1 opacity-50" />
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 2. TEST SUITE */}
            {activeTab === 'tests' && (
              <div className="h-full space-y-4">
                <div className="flex items-center justify-between p-3 bg-blue-900/20 border border-blue-700/30 rounded">
                  <span className="text-sm text-blue-400 flex items-center">
                    <Eye className="w-4 h-4 mr-2" /> Visible Test Cases ({problemData.test_cases?.length || 0})
                  </span>
                  <button onClick={() => fetchProblemDetails(true)} className="text-xs text-blue-300 hover:text-white flex items-center">
                    <RefreshCw className="w-3 h-3 mr-1" /> Refresh
                  </button>
                </div>
                <div className="space-y-3">
                  {problemData.test_cases?.map((test, index) => (
                    <div key={index} className="bg-gray-800 p-4 rounded border border-gray-700 font-mono text-sm">
                      <div className="text-gray-500 text-xs mb-2">Test Case #{index + 1}</div>
                      <div className="mb-2"><span className="text-gray-500">Input: </span><span className="text-blue-300">{JSON.stringify(test.input)}</span></div>
                      <div><span className="text-gray-500">Expected: </span><span className="text-green-300">{JSON.stringify(test.expected)}</span></div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 3. REFERENCE SOLUTION */}
            {activeTab === 'solution' && (
              <div className="h-full">
                {!hasUnlockedSolution ? (
                  <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                    <Lock className="w-12 h-12 text-gray-600" />
                    <h3 className="text-xl font-bold">Reference Solution Locked</h3>
                    <p className="text-gray-400 max-w-sm text-sm">Unlocking this will reduce your possible score to 0%.</p>
                    <button onClick={unlockSolution} className="px-6 py-2 bg-red-600 hover:bg-red-700 rounded font-bold">Agree & View</button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="p-3 bg-red-900/20 border border-red-700/30 rounded text-red-500 text-sm">Unlock Successful - Max Score: 0</div>
                    <pre className="bg-gray-800 p-4 rounded text-sm text-blue-300 font-mono overflow-x-auto">
                      {problemData.reference_solution || "# No reference solution provided"}
                    </pre>
                  </div>
                )}
              </div>
            )}

            {/* 4. WRONG REFERENCE SOLUTION (CHALLENGE AI) */}
            {activeTab === 'wrong-ref' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-xl font-bold text-white">Challenge The AI</h3>
                  <p className="text-gray-400 text-sm mt-1">If our Reference Solution fails on valid inputs, provide them here. The Gatekeeper will verify your claim, and if valid, the system will self-heal.</p>
                </div>

                {/* Input Area */}
                <div className="space-y-4">
                  {wrongCases.map((caseItem, idx) => (
                    <div key={idx} className="border border-gray-700 rounded-lg bg-gray-800/40 overflow-hidden">
                      <div className="flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700">
                        <div className="flex items-center text-xs font-mono text-gray-400">
                          <ChevronRight className="w-3 h-3 mr-1" /> CASE #{idx + 1}
                        </div>
                        {wrongCases.length > 1 && (
                          <button onClick={() => removeCase(idx)} className="text-gray-500 hover:text-red-400 transition-colors">
                            <Trash2 className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                      <div className="p-4 space-y-4">
                        <div className="space-y-1">
                          <label className="text-[10px] font-bold text-gray-500 uppercase tracking-tighter">
                            Input <span className="text-red-500">*</span>
                          </label>
                          <textarea
                            value={caseItem.input}
                            onChange={(e) => updateCase(idx, 'input', e.target.value)}
                            className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-sm font-mono text-blue-300 focus:border-blue-500 outline-none min-h-[60px]"
                            placeholder="Input values..."
                          />
                        </div>
                        <div className="space-y-1">
                          <label className="text-[10px] font-bold text-gray-500 uppercase tracking-tighter">
                            Expected Output <span className="text-red-500">*</span> (Hint: Optional (comma-separated))
                          </label>
                          <textarea
                            value={caseItem.output}
                            onChange={(e) => updateCase(idx, 'output', e.target.value)}
                            className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-sm font-mono text-green-300 focus:border-blue-500 outline-none min-h-[60px]"
                            placeholder="Correct output..."
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Buttons */}
                <div className="flex flex-col space-y-3">
                  <button onClick={addCase} disabled={isAnalyzing} className="flex items-center justify-center py-2 border-2 border-dashed border-gray-700 rounded-lg text-gray-400 hover:border-gray-500 hover:text-gray-200 transition-all text-sm font-medium disabled:opacity-50">
                    <Plus className="w-4 h-4 mr-2" /> Add Case
                  </button>
                  <button 
                    onClick={handleAnalyze}
                    disabled={isAnalyzing || !isChallengeValid}
                    className={`flex items-center justify-center py-3 rounded font-bold transition-all shadow-lg ${
                      isAnalyzing || !isChallengeValid
                      ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                    }`}
                  >
                    {isAnalyzing ? <Loader className="w-4 h-4 mr-2 animate-spin" /> : <Send className="w-4 h-4 mr-2" />}
                    {isAnalyzing ? "System Healing in Progress..." : "Submit Challenge"}
                  </button>
                </div>

                {/* LOGS TERMINAL */}
                {(analysisLogs.length > 0 || analysisResult) && (
                  <div className="mt-6 rounded-lg overflow-hidden border border-gray-700 bg-black">
                    <div className="flex items-center px-4 py-2 bg-gray-800 border-b border-gray-700">
                      <Terminal className="w-4 h-4 text-gray-400 mr-2" />
                      <span className="text-xs font-mono text-gray-400">SYSTEM LOGS</span>
                    </div>
                    <div className="p-4 font-mono text-sm space-y-2 max-h-60 overflow-y-auto">
                      {analysisLogs.map((log, i) => (
                        <div key={i} className="flex items-start">
                          {log.error ? (
                            <span className="text-red-500 mr-2">[ERROR]</span>
                          ) : (
                            <span className="text-blue-500 mr-2">
                              {log.step ? `[STEP ${log.step}]` : '[INFO]'}
                            </span>
                          )}
                          <span className={log.error ? "text-red-400" : "text-gray-300"}>
                            {log.message || log.error}
                          </span>
                        </div>
                      ))}
                      
                      {/* Final Result Display */}
                      {analysisResult && (
                        <div className={`mt-4 p-3 rounded border ${
                          analysisResult.decision === 'NO' 
                            ? 'bg-red-900/20 border-red-700 text-red-200' 
                            : 'bg-green-900/20 border-green-700 text-green-200'
                        }`}>
                          <div className="flex items-center font-bold mb-1">
                            {analysisResult.decision === 'NO' 
                              ? <XCircle className="w-4 h-4 mr-2" /> 
                              : <CheckCircle className="w-4 h-4 mr-2" />
                            }
                            {analysisResult.status}
                          </div>
                          <div className="text-xs opacity-90">{analysisResult.reason}</div>
                          {analysisResult.new_test_count && (
                             <div className="mt-2 text-[10px] font-mono opacity-70 border-t border-green-500/30 pt-1">
                               System Updated: {analysisResult.new_test_count} new tests generated.
                             </div>
                          )}
                        </div>
                      )}
                      <div ref={logsEndRef} />
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 5. RESULTS */}
            {activeTab === 'results' && submissionResult && (
              <div className="space-y-6">
                <div className="p-6 rounded-lg border bg-gray-800 border-gray-700 flex flex-col items-center">
                  <span className="text-sm text-gray-400 uppercase">Score</span>
                  <div className="text-5xl font-bold text-white mb-2">
                    {Math.floor((submissionResult.passed / submissionResult.total) * maxPossibleScore)}
                    <span className="text-xl text-gray-500"> / {maxPossibleScore}</span>
                  </div>
                </div>
                <div className={`p-4 rounded border-l-4 ${submissionResult.passed === submissionResult.total ? 'bg-green-900/20 border-green-500' : 'bg-red-900/20 border-red-500'}`}>
                  <h4 className="font-bold">{submissionResult.passed === submissionResult.total ? 'Accepted' : 'Failed'}</h4>
                  <p className="text-sm text-gray-300">{submissionResult.passed} / {submissionResult.total} cases passed.</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT PANE: EDITOR */}
        <div className="h-full bg-[#1e1e1e]">
          <Editor
            height="100%"
            defaultLanguage="python"
            theme="vs-dark"
            value={userCode}
            onChange={(val) => setUserCode(val)}
            options={{ minimap: { enabled: false }, fontSize: 14, padding: { top: 16 } }}
          />
        </div>
      </Split>
    </div>
  );
}