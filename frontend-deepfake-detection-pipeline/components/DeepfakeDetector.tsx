import React, { useState, useRef, useCallback, useEffect, useLayoutEffect } from 'react';
import { motion, AnimatePresence, useMotionValue, animate } from 'framer-motion';

// --- TYPES AND CONSTANTS ---

type Verdict = 'Authentic' | 'Deepfake';
type Status = 'idle' | 'analyzing' | 'complete';

interface ProgressReport {
  stageIndex: number;
  stageName: string;
  percent: number;
}

interface AnalysisResult {
  verdict: Verdict;
  confidence: number;
}

interface DeepfakeDetectorProps {
  onResult: (result: AnalysisResult) => void;
  simulate?: boolean;
  apiUrl?: string;
  stageDurations?: number[];
  className?: string;
}

const STAGES = [
  { name: 'Frame Extraction', icon: (p: any) => <path {...p} d="M4 4h16v16H4V4zm2 4v8h12V8H6z" /> },
  { name: 'Compression Artifact Analysis', icon: (p: any) => <path {...p} d="M10 8h4v8h-4V8zM4 8h4v8H4V8zm12 0h4v8h-4V8z" /> },
  { name: 'Facial Landmark Tracking', icon: (p: any) => <g {...p}><circle cx="12" cy="12" r="3" /><path d="M12 2a10 10 0 00-4.47 18.25m8.94 0A10 10 0 0012 2" /></g> },
  { name: 'Audio-Visual Consistency Check', icon: (p: any) => <path {...p} d="M7 15V9a5 5 0 0110 0v6m-5-4v4m-5 0h10" /> },
  { name: 'Final Classification', icon: (p: any) => <path {...p} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /> },
];

const DEFAULT_STAGE_DURATIONS = [1.2, 1.5, 2.0, 1.8, 1.0]; // in seconds

const DESKTOP_ZIGZAG = {
  path: "M 40 100 L 220 40 L 400 160 L 580 40 L 760 100",
  nodes: [
    { cx: 40, cy: 100 }, { cx: 220, cy: 40 }, { cx: 400, cy: 160 },
    { cx: 580, cy: 40 }, { cx: 760, cy: 100 },
  ],
  viewBox: "0 0 800 200",
};

const MOBILE_ZIGZAG = {
  path: "M 50 50 L 250 125 L 50 200 L 250 275 L 50 350",
  nodes: [
    { cx: 50, cy: 50 }, { cx: 250, cy: 125 }, { cx: 50, cy: 200 },
    { cx: 250, cy: 275 }, { cx: 50, cy: 350 },
  ],
  viewBox: "0 0 300 400",
};


// --- MOCK ANALYSIS FUNCTION ---

/**
 * [STUB] This function simulates a video analysis process.
 * Replace its content with your actual backend API call.
 * 
 * @param file - The video file to be analyzed.
 * @param onProgress - A callback to report progress updates.
 * @returns A promise that resolves with the analysis result.
 */
async function analyzeVideo(
  file: File,
  onProgress: (report: ProgressReport) => void,
  stageDurations: number[]
): Promise<AnalysisResult> {
  console.log(`[Stub] Starting analysis for: ${file.name}`);

  const totalDuration = stageDurations.reduce((a, b) => a + b, 0);
  let elapsedDuration = 0;

  // This simulates progress events coming from a backend.
  for (let i = 0; i < STAGES.length; i++) {
    await new Promise(resolve => setTimeout(resolve, stageDurations[i] * 1000));
    elapsedDuration += stageDurations[i];
    const percent = (elapsedDuration / totalDuration) * 100;
    onProgress({
      stageIndex: i,
      stageName: STAGES[i].name,
      percent: i === STAGES.length - 1 ? 100 : percent,
    });
  }
  
  // Simulate final result
  const isDeepfake = Math.random() > 0.5;
  return {
    verdict: isDeepfake ? 'Deepfake' : 'Authentic',
    confidence: 85 + Math.random() * 15,
  };
}


// --- MAIN COMPONENT ---

const DeepfakeDetector: React.FC<DeepfakeDetectorProps> = ({
  onResult,
  simulate = true,
  apiUrl,
  stageDurations = DEFAULT_STAGE_DURATIONS,
  className = '',
}) => {
  const [status, setStatus] = useState<Status>('idle');
  const [progress, setProgress] = useState(0);
  const [stageIndex, setStageIndex] = useState(-1);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  
  const resultCtaRef = useRef<HTMLButtonElement>(null);
  const pathRef = useRef<SVGPathElement>(null);
  const [pathLength, setPathLength] = useState(0);

  const dotProgress = useMotionValue(0);

  // --- HOOKS ---

  const checkIsMobile = useCallback(() => {
    setIsMobile(window.innerWidth < 768);
  }, []);

  useLayoutEffect(() => {
    checkIsMobile();
    window.addEventListener('resize', checkIsMobile);
    return () => window.removeEventListener('resize', checkIsMobile);
  }, [checkIsMobile]);

  useLayoutEffect(() => {
    const measure = () => {
      if (pathRef.current) {
        try {
          const len = pathRef.current.getTotalLength();
          setPathLength(len);
        } catch (e) {
          setPathLength(0);
        }
      }
    };

    // measure on next animation frame (ensures SVG is fully rendered)
    const id = requestAnimationFrame(measure);

    return () => cancelAnimationFrame(id);
  }, [isMobile, status]);


  useEffect(() => {
    if (status === 'complete' && resultCtaRef.current) {
      resultCtaRef.current.focus();
    }
  }, [status]);
  

  // --- HANDLERS ---

  /**
   * Combined handler:
   * - If `simulate` is true: run visual simulation (analyzeVideo) for UI.
   * - If `apiUrl` or VITE_API_URL is configured: upload & poll the API in parallel.
   * - API result is buffered and will be shown *after* visual simulation completes.
   * - If simulate === false, API result is shown immediately.
   */
  const handleFileSelect = useCallback(async (file: File | null) => {
    if (!file) return;

    setStatus('analyzing');
    setFileName(file.name);

    const onProgress = (report: ProgressReport) => {
      setProgress(report.percent);
      setStageIndex(report.stageIndex);

      // Prefer a continuous percent-based dot movement (0..1)
      const percent = Number(report.percent) || 0;
      const targetDotProgress = Math.max(0, Math.min(1, percent / 100)); // normalized 0..1

      animate(dotProgress, targetDotProgress, {
        duration: Math.max(0.2, (stageDurations[report.stageIndex] || 1) * 0.6),
        ease: [0.22, 1, 0.36, 1],
      });
    };

    // Helper to check for configured base URL
    const getBaseUrl = () => {
      return apiUrl || ((import.meta as any).env?.VITE_API_URL ?? '');
    };

    // Start visual simulation promise (only if simulate === true)
    let visualPromise: Promise<AnalysisResult> | null = null;
    if (simulate) {
      visualPromise = (async () => {
        onProgress({ stageIndex: -1, stageName: 'Initializing', percent: 0 });
        const simulatedResult = await analyzeVideo(file, onProgress, stageDurations);
        return simulatedResult;
      })();
    }

    // Start API upload/poll in parallel if configured
    const base = getBaseUrl();
    let apiPromise: Promise<AnalysisResult> | null = null;

    if (base) {
      apiPromise = (async () => {
        try {
          onProgress({ stageIndex: -1, stageName: 'Uploading', percent: 0 });

          const fd = new FormData();
          fd.append('file', file);

          const uploadRes = await fetch(`${base.replace(/\/$/, '')}/analyze`, {
            method: 'POST',
            body: fd,
          });

          if (!uploadRes.ok) {
            const txt = await uploadRes.text().catch(() => uploadRes.statusText || 'upload failed');
            throw new Error(`Upload failed: ${uploadRes.status} ${txt}`);
          }

          const payload = await uploadRes.json().catch(() => ({}));

          // If server returned final result inside a `result` field, use it
          if (payload.result && (payload.result.verdict || payload.result.confidence != null)) {
            const finalFromResult = payload.result as AnalysisResult;
            return {
              verdict: finalFromResult.verdict === 'Deepfake' ? 'Deepfake' : 'Authentic',
              confidence: Number(finalFromResult.confidence ?? 0),
            } as AnalysisResult;
          }

          // If server returned final result directly
          if (payload.verdict) {
            const final = payload as AnalysisResult;
            return {
              verdict: final.verdict === 'Deepfake' ? 'Deepfake' : 'Authentic',
              confidence: Number(final.confidence ?? 0),
            } as AnalysisResult;
          }

          // If backend returned an analysisId, poll a status endpoint
          if (payload.analysisId) {
            const id = payload.analysisId as string;
            let attempts = 0;
            while (attempts < 120) { // ~2 minutes
              const statusRes = await fetch(`${base.replace(/\/$/, '')}/status/${id}`);
              if (!statusRes.ok) throw new Error('Status check failed');
              const statusData = await statusRes.json().catch(() => ({}));

              if (statusData.percent != null) {
                const percent = Number(statusData.percent);
                const stageIdx = Math.min(STAGES.length - 1, Math.floor(percent / (100 / STAGES.length)));
                onProgress({ stageIndex: stageIdx, stageName: statusData.stageName ?? STAGES[stageIdx].name, percent });
              }

              if (statusData.status === 'complete' || statusData.result) {
                const final = (statusData.result ?? statusData) as AnalysisResult;
                return {
                  verdict: final.verdict === 'Deepfake' ? 'Deepfake' : 'Authentic',
                  confidence: Number(final.confidence ?? 0),
                } as AnalysisResult;
              }

              await new Promise(r => setTimeout(r, 1000));
              attempts++;
            }
            throw new Error('Analysis timed out');
          }

          // If server provided progress inside initial response
          if (payload.progress != null) {
            const percent = Number(payload.progress);
            const stageIdx = Math.min(STAGES.length - 1, Math.floor(percent / (100 / STAGES.length)));
            onProgress({ stageIndex: stageIdx, stageName: payload.stageName ?? STAGES[stageIdx].name, percent });
            if (percent >= 100 && payload.result) {
              const final = payload.result as AnalysisResult;
              return {
                verdict: final.verdict === 'Deepfake' ? 'Deepfake' : 'Authentic',
                confidence: Number(final.confidence ?? 0),
              } as AnalysisResult;
            }
          }

          // Unknown payload shape — return a fallback neutral result
          return { verdict: 'Authentic', confidence: 0 } as AnalysisResult;
        } catch (err) {
          console.error('API analysis error', err);
          throw err;
        }
      })();
    }

    // New logic:
    // - If simulation is running, we will always wait for visualPromise to finish before showing final result.
    // - API result, if available earlier, will be buffered and then applied after visual finishes.
    // - If simulate === false (visualPromise is null), API result is applied immediately (or fallback).

    let bufferedApiResult: AnalysisResult | null = null;
    let apiErrored = false;

    // Handle API resolution: buffer the result (don't set UI immediately if visual is running)
    if (apiPromise) {
      apiPromise.then((apiResult) => {
        bufferedApiResult = apiResult;
        // If no visual is running (simulate === false), show immediately
        if (!visualPromise) {
          setResult(apiResult);
          setStatus('complete');
          onResult(apiResult);
        } else {
          // visual is running — we wait for visual to finish. Optionally we could set intermediate progress/state.
          // Do nothing now; visualPromise resolution will apply bufferedApiResult.
        }
      }).catch((err) => {
        apiErrored = true;
        console.warn('API failed or errored (will rely on simulation/fallback):', err);
        // If no visual is running, show fallback now
        if (!visualPromise) {
          const fallback: AnalysisResult = { verdict: 'Authentic', confidence: 0 };
          setResult(fallback);
          setStatus('complete');
          onResult(fallback);
        }
      });
    }

    // Handle visual resolution: always show visual result when it completes,
    // BUT if API returned earlier we show the API result instead (buffered).
    if (visualPromise) {
      visualPromise.then((visualResult) => {
        // If API returned and buffered, prefer buffered API result.
        if (bufferedApiResult) {
          setResult(bufferedApiResult);
          setStatus('complete');
          onResult(bufferedApiResult);
        } else if (apiErrored && !bufferedApiResult) {
          // API errored and didn't provide result — still show visual result.
          setResult(visualResult);
          setStatus('complete');
          onResult(visualResult);
        } else {
          // API hasn't arrived yet: show visual result.
          setResult(visualResult);
          setStatus('complete');
          onResult(visualResult);
        }
      }).catch((err) => {
        console.error('Visual simulation error:', err);
        // If visual fails but API already has buffered result, show it
        if (bufferedApiResult) {
          setResult(bufferedApiResult);
          setStatus('complete');
          onResult(bufferedApiResult);
        } else if (apiErrored) {
          const fallback: AnalysisResult = { verdict: 'Authentic', confidence: 0 };
          setResult(fallback);
          setStatus('complete');
          onResult(fallback);
        } else {
          // Wait for API if available, otherwise fallback
          if (apiPromise) {
            apiPromise.then((apiResult) => {
              setResult(apiResult);
              setStatus('complete');
              onResult(apiResult);
            }).catch(() => {
              const fallback: AnalysisResult = { verdict: 'Authentic', confidence: 0 };
              setResult(fallback);
              setStatus('complete');
              onResult(fallback);
            });
          } else {
            const fallback: AnalysisResult = { verdict: 'Authentic', confidence: 0 };
            setResult(fallback);
            setStatus('complete');
            onResult(fallback);
          }
        }
      });
    } else {
      // No visual running (simulate === false)
      // If API exists, it is already handled in apiPromise.then above to show result immediately.
      // If no API, show fallback immediately.
      if (!apiPromise) {
        const fallback: AnalysisResult = { verdict: 'Authentic', confidence: 0 };
        setResult(fallback);
        setStatus('complete');
        onResult(fallback);
      }
    }

  }, [simulate, onResult, stageDurations, dotProgress, apiUrl]);

  const resetState = () => {
    setStatus('idle');
    setProgress(0);
    setStageIndex(-1);
    setResult(null);
    setFileName(null);
    dotProgress.set(0);
  };
  
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragEvents = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const { path, nodes, viewBox } = isMobile ? MOBILE_ZIGZAG : DESKTOP_ZIGZAG;

  // --- RENDER ---
  return (
    <>
      <style>{`
        :root {
          --navy: #0a192f;
          --light-navy: #112240;
          --lightest-navy: #233554;
          --slate: #8892b0;
          --light-slate: #a8b2d1;
          --lightest-slate: #ccd6f6;
          --accent: #64ffda;
          --accent-dark: #133131;
        }
        .glow {
          filter: drop-shadow(0 0 6px var(--accent));
        }
        .breathing {
          animation: breathing 2s ease-in-out infinite;
        }
        @keyframes breathing {
          0%, 100% { filter: drop-shadow(0 0 8px var(--accent)); transform: scale(1.12); }
          50% { filter: drop-shadow(0 0 12px var(--accent)); transform: scale(1.15); }
        }
      `}</style>
      <div className={`relative bg-light-navy/50 backdrop-blur-sm rounded-lg p-6 min-h-[300px] flex flex-col justify-center items-center transition-all duration-300 ${className}`}>
        <AnimatePresence mode="wait">
          {status === 'idle' && (
            <motion.div
              key="idle"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="w-full"
            >
              <Dropzone 
                onDrop={handleDrop} 
                onDragOver={handleDragEvents} 
                onDragEnter={handleDragEvents}
                onDragLeave={handleDragEvents}
                onFileChange={handleFileChange}
                isDragging={isDragging}
              />
            </motion.div>
          )}

          {status === 'analyzing' && (
            <motion.div
              key="analyzing"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="w-full flex flex-col items-center"
            >
              <div aria-live="polite" className="text-center mb-4 h-12 flex flex-col justify-center">
                <p className="text-lightest-slate font-medium text-lg">
                  {stageIndex > -1 ? STAGES[stageIndex].name : "Initializing Analysis..."}
                </p>
                <p className="text-slate text-sm font-mono truncate max-w-xs md:max-w-md">{fileName}</p>
              </div>

              <div className="relative w-full max-w-3xl aspect-[4/1] md:aspect-auto">
                <svg
                  viewBox={viewBox}
                  className="w-full h-full"
                  preserveAspectRatio="xMidYMid meet"
                >
                  <path
                    ref={pathRef}
                    d={path}
                    fill="none"
                    stroke="var(--lightest-navy)"
                    strokeWidth="2"
                    strokeDasharray="4 4"
                  />
                  {pathLength > 0 && <AnimatedDot pathRef={pathRef} pathLength={pathLength} progress={dotProgress} />}
                  
                  {nodes.map((node, i) => (
                    <PipelineNode 
                      key={i}
                      {...node}
                      label={STAGES[i].name}
                      Icon={STAGES[i].icon}
                      isActive={stageIndex >= i}
                      isMobile={isMobile}
                    />
                  ))}
                </svg>
              </div>
              
              <div className="w-full max-w-sm mt-4">
                  <div className="w-full bg-lightest-navy/50 rounded-full h-2.5">
                      <motion.div
                          className="bg-accent h-2.5 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${progress}%` }}
                          transition={{ type: 'spring', stiffness: 50, damping: 20 }}
                      />
                  </div>
                  <p className="text-center mt-2 font-mono text-accent">{progress.toFixed(0)}%</p>
              </div>
            </motion.div>
          )}

          {status === 'complete' && result && (
             <motion.div
                key="complete"
                initial={{ opacity: 0, y: 30, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
                className="text-center"
            >
                <ResultCard result={result} onReset={resetState} ctaRef={resultCtaRef} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
};


// --- SUB-COMPONENTS ---

const Dropzone: React.FC<any> = ({ onDrop, onDragOver, onDragEnter, onDragLeave, onFileChange, isDragging }) => {
    const inputRef = useRef<HTMLInputElement>(null);
    return (
        <div
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragEnter={onDragEnter}
            onDragLeave={onDragLeave}
            className={`w-full h-full p-8 border-2 border-dashed rounded-lg flex flex-col justify-center items-center cursor-pointer transition-colors duration-300 ${isDragging ? 'border-accent bg-accent-dark' : 'border-lightest-navy hover:border-accent hover:bg-lightest-navy/10'}`}
            onClick={() => inputRef.current?.click()}
        >
            <input
                ref={inputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={onFileChange}
            />
             <svg xmlns="http://www.w3.org/2000/svg" className={`w-16 h-16 mb-4 transition-colors ${isDragging ? 'text-accent' : 'text-light-slate'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 16a4 4 0 01-4-4V7a4 4 0 014-4h10a4 4 0 014 4v5a4 4 0 01-4 4H7z" /><polyline points="12 8 12 16" /><polyline points="8 12 12 8 16 12" /></svg>
            <p className="text-lightest-slate font-semibold">Drag & drop a video file here</p>
            <p className="text-slate mt-1">or</p>
            <button
                type="button"
                className="mt-4 px-6 py-2 bg-accent text-navy font-bold rounded hover:bg-opacity-80 transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-light-navy focus:ring-accent"
            >
                Browse Files
            </button>
        </div>
    );
};

const PipelineNode: React.FC<any> = ({ cx, cy, label, Icon, isActive, isMobile }) => {
  const textY = isMobile ? (label.includes('Facial') || label.includes('Final') ? cy + 30 : cy + 25) : (cy > 100 ? cy + 40 : cy - 30);
  const textAnchor = isMobile ? (cx < 150 ? 'end' : 'start') : 'middle';
  const textX = isMobile ? (cx < 150 ? cx - 20 : cx + 20) : cx;

  return (
    <motion.g
        initial={{ scale: 1, opacity: 0.7 }}
        animate={isActive ? "active" : "inactive"}
        variants={{
            active: { scale: 1.12, opacity: 1, transition: { duration: 0.22 } },
            inactive: { scale: 1, opacity: 0.7 }
        }}
    >
      <circle cx={cx} cy={cy} r="16" fill="var(--light-navy)" stroke={isActive ? "var(--accent)" : "var(--lightest-navy)"} strokeWidth="2" />
      <Icon fill={isActive ? "var(--accent)" : "var(--slate)"} transform={`translate(${cx-12}, ${cy-12})`} />
      <motion.circle 
          cx={cx} 
          cy={cy} 
          r="16" 
          fill="none" 
          stroke="var(--accent)" 
          strokeWidth="3" 
          className={isActive ? 'breathing' : ''}
          initial={{ opacity: 0 }}
          animate={{ opacity: isActive ? 1 : 0 }}
      />
      <text
        x={textX}
        y={textY}
        textAnchor={textAnchor}
        fontSize={isMobile ? "12" : "14"}
        className="transition-fill duration-300"
        fill={isActive ? "var(--lightest-slate)" : "var(--slate)"}
        style={{ fontWeight: isActive ? 600 : 400 }}
      >
        {label}
      </text>
    </motion.g>
  );
};


const AnimatedDot = ({ pathRef, pathLength, progress }: { pathRef: React.RefObject<SVGPathElement>, pathLength: number, progress: any }) => {
    const [pos, setPos] = useState({ x: 0, y: 0 });

    useEffect(() => {
        return progress.on("change", (latest: number) => {
            if (pathRef.current && pathLength > 0) {
                const point = pathRef.current.getPointAtLength(latest * pathLength);
                setPos({ x: point.x, y: point.y });
            }
        });
    }, [pathRef, pathLength, progress]);

    return (
      <g transform={`translate(${pos.x}, ${pos.y})`}>
        <circle r="8" fill="var(--accent)" className="glow" />
        <motion.circle
          r="16"
          fill="var(--accent)"
          initial={{ scale: 1, opacity: 0.3 }}
          animate={{ scale: 2, opacity: 0 }}
          transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
        />
      </g>
    );

};

const ResultCard: React.FC<{ result: AnalysisResult, onReset: () => void, ctaRef: React.RefObject<HTMLButtonElement> }> = ({ result, onReset, ctaRef }) => {
  const isDeepfake = result.verdict === 'Deepfake';
  const Icon = isDeepfake 
    ? <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    : <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />;

  return (
    <div className="flex flex-col items-center">
      <svg xmlns="http://www.w3.org/2000/svg" className={`w-24 h-24 mb-4 ${isDeepfake ? 'text-yellow-400' : 'text-accent'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">{Icon}</svg>
      <h2 className={`text-2xl font-bold tracking-wider ${isDeepfake ? 'text-yellow-400' : 'text-accent'}`}>{result.verdict.toUpperCase()}{isDeepfake ? '' : ' MEDIA'}</h2>
      <p className="text-lightest-slate mt-2">Confidence Score: <span className="font-bold text-white">{result.confidence.toFixed(2)}%</span></p>
      <button
        ref={ctaRef}
        onClick={onReset}
        className="mt-8 px-8 py-3 bg-accent text-navy font-bold rounded hover:bg-opacity-80 transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-light-navy focus:ring-accent"
      >
        Analyze Another Video
      </button>
    </div>
  );
};


export default DeepfakeDetector;

/*
// --- EXAMPLE USAGE & BACKEND INTEGRATION ---

Usage notes:
- Keep `simulate={true}` in App.tsx if you want the UI simulation and still have the API run in background.
- Provide `apiUrl="http://localhost:8000"` (or set `VITE_API_URL`) to enable background upload + polling.
- API endpoint expectations:
  POST /analyze   (multipart form file field "file") -> returns either:
    - { result: { verdict: "Deepfake"|"Authentic", confidence: number } }
    - OR { verdict: "...", confidence: ... }
    - OR { analysisId: "..." } (then /status/:id returns { percent, stageName, status, result? })
*/
