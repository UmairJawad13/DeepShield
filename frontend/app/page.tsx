"use client";

import { useState, useRef, useEffect } from "react";
import { Upload, Shield, Activity, Search, AlertTriangle, ShieldCheck, FileImage, Info, History, Clock } from "lucide-react";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

const generateThumbnail = (file: File): Promise<string> => {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        const MAX_WIDTH = 120;
        const scaleSize = MAX_WIDTH / img.width;
        canvas.width = MAX_WIDTH;
        canvas.height = img.height * scaleSize;
        const ctx = canvas.getContext("2d");
        ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.7));
      };
      img.src = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  });
};

const formatTimeAgo = (timestamp: number) => {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return `${seconds} secs ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} mins ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hrs ago`;
  return `${Math.floor(hours / 24)} days ago`;
};

export default function DeepShieldDashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // States: 'idle' | 'uploading' | 'processing' | 'results' | 'error'
  const [status, setStatus] = useState<string>("idle");
  const [progress, setProgress] = useState(0);

  const [result, setResult] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    const saved = localStorage.getItem("deepshield_history");
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) { }
    }
  }, []);

  const addToHistory = (resultItem: any, thumbnailBase64: string) => {
    setHistory((prev) => {
      const newEntry = {
        id: Date.now().toString(),
        timestamp: Date.now(),
        result: resultItem,
        thumbnail: thumbnailBase64
      };
      const updated = [newEntry, ...prev].slice(0, 10);
      localStorage.setItem("deepshield_history", JSON.stringify(updated));
      return updated;
    });
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (selectedFile: File) => {
    if (!selectedFile.type.startsWith("image/") && !selectedFile.type.startsWith("video/mp4") && !selectedFile.type.startsWith("video/quicktime") && !selectedFile.name.endsWith(".mov")) {
      alert("Please upload a valid image or video (.mp4, .mov) file.");
      return;
    }
    setFile(selectedFile);

    // For video, we might not always have a neat preview immediately from ObjectURL if it's large, but standard video tags can use it.
    // For now, we will use it
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setStatus("idle");
    setResult(null);
  };

  const analyzeImage = async () => {
    if (!file) return;
    setStatus("uploading");
    setProgress(10);

    try {
      const formData = new FormData();
      formData.append("file", file);

      setProgress(30);

      const apiUrl = "https://subcoriaceous-apolitically-neely.ngrok-free.dev";
      const response = await fetch(`${apiUrl}/api/detect`, {
        method: "POST",
        headers: {
          "ngrok-skip-browser-warning": "69420"
        },
        body: formData,
      });

      if (!response.ok) throw new Error("Upload Failed");
      const initData = await response.json();

      // Handle the Bypass Mode JSON immediately
      if (initData.status === "success") {
        setProgress(100);
        const bypassResult = {
          is_fake: initData.is_deepfake,
          confidence: initData.confidence * 100, // Converts 0.99 to 99
          heatmap_url: "",
          metadata: { model: "Bypass Validation", analysis: { camera_make: "Mock Data" } }
        };
        setResult(bypassResult);
        setTimeout(() => setStatus("results"), 500);
        return;
      }

      const taskId = initData.task_id;

      setStatus("processing");
      setProgress(50);

      // Poll for background task completion
      const pollInterval = setInterval(async () => {
        const apiUrl = "https://subcoriaceous-apolitically-neely.ngrok-free.dev";
        const pollRes = await fetch(`${apiUrl}/api/detect/${taskId}`, {
          headers: {
            "ngrok-skip-browser-warning": "69420"
          }
        });
        if (!pollRes.ok) return;

        const pollData = await pollRes.json();

        if (pollData.status === "completed") {
          clearInterval(pollInterval);
          setProgress(100);
          setResult(pollData.result);
          if (file) {
            generateThumbnail(file).then((thumb) => {
              addToHistory(pollData.result, thumb);
            });
          }
          setTimeout(() => setStatus("results"), 500); // Small delay for UI polish
        } else if (pollData.status === "failed") {
          clearInterval(pollInterval);
          setStatus("error");
          console.error(pollData.error);
        } else {
          // processing - increment visual progress artificially
          setProgress((prev) => Math.min(prev + 10, 90));
        }
      }, 1000);

    } catch (err) {
      setStatus("error");
      console.error(err);
    }
  };

  const reset = () => {
    setFile(null);
    setPreviewUrl(null);
    setStatus("idle");
    setResult(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col relative overflow-hidden font-[family-name:var(--font-outfit)]">
      {/* Background Cyber Tech Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(0,240,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,240,255,0.03)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary to-transparent opacity-20" />

      {/* Header */}
      <header className="relative z-10 flex items-center justify-between px-8 py-6 border-b border-border/40 glass-panel">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10 border border-primary/20">
            <Shield className="w-6 h-6 text-primary" />
          </div>
          <h1 className="text-2xl font-bold tracking-wider text-white">DEEP<span className="text-primary">SHIELD</span></h1>
        </div>
        <div className="flex items-center gap-4 text-sm text-muted-foreground font-mono">
          <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-black/40 border border-primary/20 rounded-full text-xs">
            <span className="text-primary font-bold">Architecture:</span> EfficientNet-B4 | <span className="text-primary font-bold">Engine:</span> PyTorch | <span className="text-primary font-bold">XAI:</span> Grad-CAM
          </div>
          <span className="flex items-center gap-2 tracking-widest font-bold">
            <span className="relative flex h-2 w-2">
              <span className="animate-pulse absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
            </span>
            SYSTEM: ONLINE
          </span>

          <Sheet>
            <SheetTrigger asChild>
              <Button variant="outline" size="sm" className="bg-black/50 border-primary/30 text-white hover:bg-primary/20 hover:text-primary transition-all">
                <History className="w-4 h-4 mr-2" />
                History
              </Button>
            </SheetTrigger>
            <SheetContent className="bg-black/80 backdrop-blur-xl border-l border-primary/20 text-white w-full sm:max-w-md overflow-y-auto">
              <SheetHeader>
                <SheetTitle className="text-white text-2xl font-bold flex items-center gap-2">
                  <Clock className="w-5 h-5 text-primary" /> Forensic Log
                </SheetTitle>
                <SheetDescription className="text-gray-400">
                  Recent platform scans. Limited to 10 entries locally.
                </SheetDescription>
              </SheetHeader>
              <div className="mt-8 flex flex-col gap-4">
                {history.length === 0 ? (
                  <p className="text-gray-500 text-sm text-center py-10">No recent scans found.</p>
                ) : (
                  history.map((item) => (
                    <div
                      key={item.id}
                      onClick={() => {
                        setResult(item.result);
                        setPreviewUrl(item.thumbnail);
                        setStatus("results");
                      }}
                      className="group flex items-center justify-between p-3 rounded-lg border border-white/10 hover:border-primary/50 bg-white/5 hover:bg-black transition-all cursor-pointer shadow-sm hover:shadow-[0_0_15px_rgba(0,240,255,0.2)]"
                    >
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-full overflow-hidden border border-white/20 group-hover:border-primary shrink-0 transition-colors bg-black/50">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={item.thumbnail} alt="scan thumbnail" className="w-full h-full object-cover" />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className={`text-sm font-bold tracking-wider ${item.result.is_fake ? 'text-red-500 [text-shadow:0_0_5px_rgba(239,68,68,0.5)]' : 'text-cyan-400 [text-shadow:0_0_5px_rgba(34,211,238,0.5)]'}`}>
                            {item.result.is_fake ? 'FAKE' : 'REAL'}
                          </span>
                          <span className="text-xs text-gray-500 font-mono">
                            {formatTimeAgo(item.timestamp)}
                          </span>
                        </div>
                      </div>
                      <div className="text-right flex flex-col items-end">
                        <span className="text-xs text-gray-400">Score</span>
                        <span className="text-sm font-mono font-bold text-white max-w-24 truncate">{item.result.confidence}%</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 relative z-10 max-w-5xl mx-auto w-full p-6 lg:p-12 flex flex-col items-center justify-center">

        <AnimatePresence mode="wait">

          {/* Default Upload View */}
          {(status === "idle" || status === "uploading") && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, y: -20 }}
              className="w-full max-w-2xl"
            >
              <Card className="glass-panel-neon border-primary/30">
                <CardHeader className="text-center pb-2">
                  <CardTitle className="text-3xl font-bold tracking-tight text-white mb-2">Upload File for Analysis</CardTitle>
                  <CardDescription className="text-base">Ensure the image contains a clear view of the subject's face.</CardDescription>
                </CardHeader>
                <CardContent className="pt-6">
                  {/* Dropzone */}
                  {!file ? (
                    <div
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                      className="border-2 border-dashed border-primary/30 rounded-xl p-12 flex flex-col items-center justify-center gap-4 hover:border-primary/60 transition-colors cursor-pointer bg-black/20 group"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <input
                        type="file"
                        ref={fileInputRef}
                        className="hidden"
                        accept="image/*,video/mp4,video/quicktime,.mov"
                        onChange={handleFileSelect}
                      />
                      <div className="p-4 rounded-full bg-primary/10 group-hover:bg-primary/20 transition-colors">
                        <Upload className="w-8 h-8 text-primary" />
                      </div>
                      <div className="text-center">
                        <p className="font-semibold text-white mb-1">Drag & Drop or Click to Browse</p>
                        <p className="text-sm text-muted-foreground">Supports JPG, PNG, MP4, MOV (Max 50MB)</p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-6">
                      <div className="relative border border-primary/20 rounded-xl overflow-hidden bg-black/50 aspect-video flex items-center justify-center p-4">
                        {file && file.type.startsWith("video/") ? (
                          <video src={previewUrl as string} className="max-h-full object-contain rounded-md shadow-lg" controls />
                        ) : (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={previewUrl as string} alt="Preview" className="max-h-full object-contain rounded-md shadow-lg" />
                        )}
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <FileImage className="text-primary w-5 h-5" />
                          <span className="text-sm text-gray-300 font-mono truncate max-w-[200px]">{file.name}</span>
                        </div>
                        <div className="flex gap-3">
                          <Button variant="outline" className="border-primary/30 hover:bg-primary/10 text-white" onClick={() => setFile(null)}>Cancel</Button>
                          <Button className="bg-primary text-black hover:bg-primary/80 font-semibold tracking-wide" onClick={analyzeImage} disabled={status !== "idle"}>
                            {status === "uploading" ? "Initiating..." : "Analyze Media"}
                          </Button>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Error View */}
          {status === "error" && (
            <motion.div
              key="error"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, y: -20 }}
              className="w-full max-w-2xl"
            >
              <Card className="glass-panel-neon border-destructive/50">
                <CardHeader className="text-center pb-2">
                  <AlertTriangle className="w-12 h-12 text-destructive mx-auto mb-2" />
                  <CardTitle className="text-3xl font-bold tracking-tight text-white mb-2">Analysis Failed</CardTitle>
                  <CardDescription className="text-base text-red-400">An unexpected system error occurred during inference.</CardDescription>
                </CardHeader>
                <CardContent className="pt-6 flex justify-center">
                  <Button onClick={reset} className="bg-white/10 hover:bg-white/20 text-white">Return to Dashboard</Button>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Processing / Scanning View */}
          {status === "processing" && (
            <motion.div
              key="processing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="w-full max-w-4xl"
            >
              <div className="flex flex-col items-center justify-center gap-8 py-12">

                {/* The WOW Factor: Scanning Image */}
                <div className="relative border-2 border-primary/50 shadow-[0_0_30px_rgba(0,240,255,0.2)] rounded-lg overflow-hidden w-full max-w-md aspect-square bg-black flex items-center justify-center">
                  {file && file.type.startsWith("video/") ? (
                    <video src={previewUrl as string} className="w-full h-full object-cover opacity-50 grayscale" autoPlay muted loop />
                  ) : (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={previewUrl as string} className="w-full h-full object-cover opacity-50 grayscale" alt="Processing" />
                  )}

                  {/* Horizontal Laser Line */}
                  <div className="absolute left-0 w-full h-[2px] bg-primary shadow-[0_0_15px_2px_rgba(0,240,255,0.8)] z-20 animate-laser"></div>

                  {/* Scanning Grid Overlay */}
                  <div className="absolute inset-0 bg-[linear-gradient(rgba(0,240,255,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(0,240,255,0.1)_1px,transparent_1px)] bg-[size:20px_20px] mix-blend-overlay z-10 pointer-events-none" />
                </div>

                <div className="w-full max-w-md space-y-4 text-center">
                  <div className="flex items-center justify-center gap-3 text-primary mb-2">
                    <Search className="w-5 h-5 animate-spin" />
                    <h3 className="text-xl font-mono tracking-widest uppercase">Forensic Analysis Active</h3>
                  </div>
                  <Progress value={progress} className="h-1" indicatorClassName="bg-primary" />
                  <p className="text-sm text-muted-foreground font-mono">Running Extracted Features via EfficientNet-B4...</p>
                </div>

              </div>
            </motion.div>
          )}

          {/* Results View */}
          {status === "results" && result && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="w-full max-w-5xl"
            >
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* Result Card */}
                <Card className={`glass-panel border ${result.is_fake ? 'border-destructive/50 shadow-[0_0_30px_rgba(255,42,42,0.15)]' : 'border-primary/50 shadow-[0_0_30px_rgba(0,240,255,0.15)]'}`}>
                  <CardHeader>
                    <div className="flex items-center gap-3 mb-2">
                      {result.is_fake ? <AlertTriangle className="w-8 h-8 text-destructive" /> : <ShieldCheck className="w-8 h-8 text-primary" />}
                      <CardTitle className="text-2xl font-bold text-white">Detection Report</CardTitle>
                    </div>
                    <CardDescription>Final diagnostic extracted from Convolutional layers</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="p-4 rounded-lg bg-black/40 border border-white/5 flex flex-col items-center justify-center gap-2 w-full">
                      <span className="text-sm text-gray-400 font-mono">CLASSIFICATION</span>
                      <h2 className={`text-2xl md:text-3xl font-bold tracking-widest uppercase text-center w-full ${result.is_fake ? 'text-red-500 [text-shadow:0_0_15px_rgba(239,68,68,0.6)]' : 'text-cyan-400 [text-shadow:0_0_15px_rgba(34,211,238,0.6)]'}`}>
                        {result.is_fake ? 'DEEPFAKE' : 'AUTHENTIC'}
                      </h2>
                    </div>

                    <div className="space-y-3 font-mono text-sm">
                      <div className="flex flex-col py-2 border-b border-white/5 gap-2">
                        <div className="flex justify-between items-center w-full">
                          <span className="text-gray-400">Confidence Score</span>
                          <span className="text-white text-base">{result.confidence}%</span>
                        </div>
                        <div className="w-full h-1.5 bg-black rounded-full overflow-hidden">
                          <div
                            className={`h-full ${result.is_fake ? 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.8)]' : 'bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.8)]'}`}
                            style={{ width: `${result.confidence}%` }}>
                          </div>
                        </div>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-white/5">
                        <span className="text-gray-400">Detection Model</span>
                        <span className="text-white text-base">{result.metadata.model}</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-white/5">
                        <span className="text-gray-400">Timestamp</span>
                        <span className="text-white text-base">{new Date().toLocaleTimeString()}</span>
                      </div>
                    </div>

                    <Button onClick={reset} className="w-full bg-white/5 text-white mt-4 border border-white/10 transition-all duration-500 hover:bg-black hover:border-primary hover:text-primary hover:shadow-[0_0_20px_rgba(0,240,255,0.4)] relative overflow-hidden group" variant="outline">
                      <span className="relative z-10">Perform New Analysis</span>
                      <span className="absolute inset-0 w-full h-full bg-primary/10 scale-x-0 group-hover:scale-x-100 origin-left transition-transform duration-500 ease-out"></span>
                    </Button>
                  </CardContent>
                </Card>

                {/* Heatmap Card */}
                <Card className="glass-panel border-primary/20">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xl text-white">Grad-CAM Explanation</CardTitle>
                      <span className="px-2 py-1 bg-primary/20 text-primary text-xs font-mono rounded border border-primary/30">HEATMAP</span>
                    </div>
                    <CardDescription>Visualizing highly manipulated regions identified by AI</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="relative rounded-lg overflow-hidden border border-white/10 bg-black aspect-square flex items-center justify-center group">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={result.heatmap_url} alt="Heatmap Explainability" className="w-full h-full object-cover" />

                      {/* Interactive overlay on hover */}
                      <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center backdrop-blur-sm">
                        <p className="text-xs text-center font-mono text-primary px-8">Red regions indicate areas that influenced the model's decision the most.</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Metadata Analysis Card */}
                <Card className="glass-panel border-primary/20">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xl text-white">Metadata Analysis</CardTitle>
                      <span className="px-2 py-1 bg-primary/20 text-primary text-xs font-mono rounded border border-primary/30">EXIF</span>
                    </div>
                    <CardDescription>File fingerprinting and structural integrity check</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4 font-mono text-sm overflow-visible">
                    {result.metadata?.analysis && (
                      <div className="space-y-3">
                        <div className="p-3 bg-black/40 border border-white/5 rounded-md flex flex-col gap-1">
                          <div className="flex justify-between border-b border-white/10 pb-1 items-center">
                            <div className="text-gray-400 flex items-center gap-1.5">
                              <span>Camera Make</span>
                              {result.metadata.analysis.camera_make === "Unknown" && (
                                <TooltipProvider>
                                  <Tooltip delayDuration={100}>
                                    <TooltipTrigger asChild>
                                      <Info className="w-4 h-4 shrink-0 text-gray-500 hover:text-primary transition-colors cursor-help" />
                                    </TooltipTrigger>
                                    <TooltipContent side="top" className="z-50 p-2 bg-black border border-white/20 text-gray-300 max-w-xs text-center shadow-lg">
                                      <p className="text-xs">Standard EXIF data is missing, which is common in AI-generated imagery.</p>
                                    </TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              )}
                            </div>
                            <span className="text-white text-right">{result.metadata.analysis.camera_make}</span>
                          </div>
                          <div className="flex justify-between border-b border-white/10 py-1">
                            <span className="text-gray-400">Camera Model</span>
                            <span className="text-white text-right">{result.metadata.analysis.camera_model}</span>
                          </div>
                          <div className="flex justify-between border-b border-white/10 py-1">
                            <span className="text-gray-400">Software</span>
                            <span className="text-white text-right break-words max-w-[150px]">{result.metadata.analysis.software}</span>
                          </div>
                          <div className="flex justify-between border-b border-white/10 py-1">
                            <span className="text-gray-400">Format</span>
                            <span className="text-white text-right">{result.metadata.analysis.format}</span>
                          </div>
                          <div className="flex justify-between py-1">
                            <span className="text-gray-400">Dimensions</span>
                            <span className="text-white text-right">{result.metadata.analysis.size}</span>
                          </div>
                        </div>

                        {result.metadata.analysis.is_suspicious && (
                          <div className="p-3 bg-destructive/20 border border-destructive/50 rounded-md">
                            <h4 className="flex items-center gap-2 text-destructive font-bold mb-2 uppercase">
                              <AlertTriangle className="w-4 h-4" /> Suspicious Markers Detected
                            </h4>
                            <ul className="list-disc list-inside text-xs text-red-200/90 space-y-1">
                              {result.metadata.analysis.suspicious_reasons.map((reason: string, idx: number) => (
                                <li key={idx}>{reason}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {!result.metadata.analysis.is_suspicious && (
                          (result.metadata.analysis.camera_make === "Unknown" || result.metadata.analysis.camera_model === "Unknown") && result.is_fake ? (
                            <div className="p-3 bg-yellow-500/20 border border-yellow-500 rounded-md">
                              <h4 className="flex items-center gap-2 text-yellow-500 font-bold uppercase">
                                <AlertTriangle className="w-4 h-4" /> Metadata Stripped
                              </h4>
                            </div>
                          ) : (
                            <div className="p-3 bg-primary/10 border border-primary/30 rounded-md">
                              <h4 className="flex items-center gap-2 text-primary font-bold uppercase">
                                <ShieldCheck className="w-4 h-4" /> Clean Metadata
                              </h4>
                            </div>
                          )
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>

              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
