import { ChevronLeft, ChevronRight, Save, Settings2, X, Box as BoxIcon, Move, Trash2, Copy, RotateCcw, LayoutGrid, Zap, Layers, ChevronDown, Search } from 'lucide-react';
import React, { useCallback, useEffect, useState, useRef, useMemo } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { API } from '../api';
import { Annotator } from '../components/canvas/Annotator';
import { Sample, Waypoint } from '../types';

const COLORS = [
  '#ff003c', '#00ff41', '#00f3ff', '#ffcc00', '#ff00ff',
  '#00ffff', '#ffff00', '#ff8800', '#88ff00', '#00ff88',
  '#0088ff', '#8800ff', '#ff0088', '#ffffff'
];

export const AnnotatePage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { id } = useParams<{ id: string }>();
  const projectId = id ? parseInt(id) : 1;
  const initialFile = searchParams.get('file');

  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedFilename, setSelectedFilename] = useState<string | null>(null);
  const [currentData, setCurrentData] = useState<Partial<Sample>>({ bboxes: [], waypoints: [], control_points: [], command: 0 });
  const [mode, setMode] = useState<'bbox' | 'waypoint'>('bbox');
  const [classes, setClasses] = useState<string[]>([]);
  const [isClassModalOpen, setIsClassModalOpen] = useState(false);
  const [history, setHistory] = useState<Partial<Sample>[]>([]);
  const [selectedBBoxIdx, setSelectedBBoxIdx] = useState<number | null>(null);
  const [filter, setFilter] = useState<'all' | 'labeled' | 'unlabeled'>(initialFile ? 'all' : 'unlabeled');
  
  // Speed Optimizations
  const [lastSelectedClass, setLastSelectedClass] = useState<number>(0);
  const [classSearch, setClassSearch] = useState('');

  const initialized = useRef(false);

  const loadClasses = useCallback(async () => {
    try {
        const list = await API.labels.getClasses(projectId);
        if (list) setClasses(list);
    } catch(e) {}
  }, [projectId]);

  const loadFiles = useCallback(async () => {
    const list = await API.labels.list({ 
        limit: 5000, 
        project_id: projectId,
        is_labeled: filter === 'all' ? undefined : (filter === 'labeled')
    });
    setSamples(list);
  }, [projectId, filter]);

  useEffect(() => {
    loadClasses();
    loadFiles();
  }, [loadClasses, loadFiles]);

  const handleSelect = useCallback(async (filename: string) => {
    setSelectedFilename(filename);
    setSelectedBBoxIdx(null);
    const data = await API.labels.get(filename);
    setCurrentData({
        ...data,
        bboxes: data.bboxes || [],
        waypoints: data.waypoints || [],
        control_points: data.control_points || []
    });
    setHistory([]);
    setSearchParams({ file: filename }, { replace: true });
  }, [setSearchParams]);

  const handleUpdate = (updated: Partial<Sample>) => {
    let finalUpdate = { ...updated };

    // Auto-resample Bezier
    if (updated.waypoints && updated.waypoints.length === 4) {
        const [p0, p1, p2, p3] = updated.waypoints;
        const resampled: Waypoint[] = [];
        for (let i = 0; i < 10; i++) {
            const t = i / 9;
            const cx = (1 - t) ** 3 * p0.x + 3 * (1 - t) ** 2 * t * p1.x + 3 * (1 - t) * t ** 2 * p2.x + t ** 3 * p3.x;
            const cy = (1 - t) ** 3 * p0.y + 3 * (1 - t) ** 2 * t * p1.y + 3 * (1 - t) * t ** 2 * p2.y + t ** 3 * p3.y;
            resampled.push({ x: cx, y: cy });
        }
        finalUpdate.waypoints = resampled;
    }

    // Auto-apply Sticky Class to NEWLY created bboxes
    if (updated.bboxes && updated.bboxes.length > (currentData.bboxes?.length || 0)) {
        const newBoxIdx = updated.bboxes.length - 1;
        if (finalUpdate.bboxes && finalUpdate.bboxes[newBoxIdx].category === 0 && lastSelectedClass !== 0) {
            finalUpdate.bboxes[newBoxIdx].category = lastSelectedClass;
        }
    }

    setHistory(prev => [JSON.parse(JSON.stringify(currentData)), ...prev].slice(0, 50));
    setCurrentData(prev => ({ ...prev, ...finalUpdate }));
  };

  const handleUndo = useCallback(() => {
    if (history.length > 0) {
        const prev = history[0];
        setCurrentData(prev);
        setHistory(prevHistory => prevHistory.slice(1));
    }
  }, [history]);

  const handleSave = async () => {
    if (!selectedFilename) return;
    await API.labels.save(selectedFilename, {
      command: currentData.command || 0,
      bboxes: currentData.bboxes || [],
      waypoints: currentData.waypoints || [],
      control_points: currentData.control_points || []
    } as any);
    setSamples(prev => prev.map(s => s.filename === selectedFilename ? { ...s, is_labeled: true } : s));
  };

  const handleSpawnTemplate = (type: 'straight' | 'left' | 'right') => {
    let p0 = {x:0.5, y:0.9}, p3 = {x:0.5, y:0.3};
    let p1 = {x:0.5, y:0.7}, p2 = {x:0.5, y:0.5};
    if (type === 'left') { p2 = {x:0.3, y:0.5}; p3 = {x:0.1, y:0.5}; }
    if (type === 'right') { p2 = {x:0.7, y:0.5}; p3 = {x:0.9, y:0.5}; }
    const ctrls = [p0, p1, p2, p3];
    const pts: Waypoint[] = [];
    for (let i = 0; i < 10; i++) {
        const t = i / 9;
        const cx = (1 - t) ** 3 * p0.x + 3 * (1 - t) ** 2 * t * p1.x + 3 * (1 - t) * t ** 2 * p2.x + t ** 3 * p3.x;
        const cy = (1 - t) ** 3 * p0.y + 3 * (1 - t) ** 2 * t * p1.y + 3 * (1 - t) * t ** 2 * p2.y + t ** 3 * p3.y;
        pts.push({ x: cx, y: cy });
    }
    handleUpdate({ control_points: ctrls, waypoints: pts });
    setMode('waypoint');
  };

  useEffect(() => {
    const handleKeys = (e: KeyboardEvent) => {
      if (document.activeElement?.tagName === 'INPUT') return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); handleUndo(); }
      if (e.key === 'ArrowRight') {
          const idx = samples.findIndex(s => s.filename === selectedFilename);
          if (idx < samples.length - 1) handleSelect(samples[idx + 1].filename);
      }
      if (e.key === 'ArrowLeft') {
          const idx = samples.findIndex(s => s.filename === selectedFilename);
          if (idx > 0) handleSelect(samples[idx - 1].filename);
      }
      if (e.key.toLowerCase() === 's') {
        if (e.ctrlKey || e.metaKey) e.preventDefault();
        handleSave();
      }
      if (e.key.toLowerCase() === 'q') setMode('bbox');
      if (e.key.toLowerCase() === 'w') setMode('waypoint');
    };
    window.addEventListener('keydown', handleKeys);
    return () => window.removeEventListener('keydown', handleKeys);
  }, [selectedFilename, mode, handleUndo, samples, handleSelect]);

  useEffect(() => {
    if (samples.length > 0 && !initialized.current) {
        if (initialFile && samples.some(s => s.filename === initialFile)) {
            handleSelect(initialFile);
        } else if (!selectedFilename) {
            handleSelect(samples[0].filename);
        }
        initialized.current = true;
    }
  }, [samples, initialFile, handleSelect, selectedFilename]);

  const filteredClasses = useMemo(() => {
    return classes.map((c, i) => ({ name: c, id: i }))
                  .filter(c => c.name.toLowerCase().includes(classSearch.toLowerCase()));
  }, [classes, classSearch]);

  const currentIndex = samples.findIndex(s => s.filename === selectedFilename);

  return (
    <div className="h-screen w-screen bg-[#050505] text-white flex flex-col overflow-hidden select-none font-bold">
      <header className="h-16 border-b border-white/20 flex items-center justify-between px-8 bg-[#0a0a0c] z-50 shadow-xl">
        <div className="flex items-center gap-8">
            <button onClick={() => navigate(`/dataset/${projectId}`)} className="text-white hover:text-accent transition-all hover:scale-110">
                <LayoutGrid className="w-6 h-6" />
            </button>
            <div className="flex flex-col">
                <h1 className="text-[11px] font-cyber tracking-[0.3em] text-accent uppercase">NeuroPilot_Extreme_v4</h1>
                <p className="text-[12px] text-white/90 font-mono font-black">{selectedFilename || 'AWAITING_INPUT...'}</p>
            </div>
        </div>

        <div className="flex items-center gap-6">
            <div className="flex bg-white/10 rounded-xl p-1 border border-white/10 shadow-inner">
                <button onClick={() => { setFilter('unlabeled'); initialized.current = false; }} className={`px-5 py-2 text-[11px] font-black rounded-lg transition-all ${filter === 'unlabeled' ? 'bg-accent text-black shadow-lg' : 'text-white/40 hover:text-white'}`}>TODO</button>
                <button onClick={() => { setFilter('all'); initialized.current = false; }} className={`px-5 py-2 text-[11px] font-black rounded-lg transition-all ${filter === 'all' ? 'bg-accent text-black shadow-lg' : 'text-white/40 hover:text-white'}`}>ALL</button>
            </div>
            <div className="flex items-center gap-3 bg-white/10 px-5 py-2 rounded-2xl border border-white/20 shadow-xl">
                <button onClick={() => { const idx = samples.findIndex(s => s.filename === selectedFilename); if (idx > 0) handleSelect(samples[idx-1].filename); }} className="p-1 hover:text-accent transition-colors"><ChevronLeft className="w-7 h-7" /></button>
                <div className="px-4 py-1.5 bg-black/60 rounded-xl text-[13px] font-black font-mono border border-white/10 text-accent">
                    {samples.length > 0 ? currentIndex + 1 : 0} <span className="text-white/20 mx-1">/</span> {samples.length}
                </div>
                <button onClick={() => { const idx = samples.findIndex(s => s.filename === selectedFilename); if (idx < samples.length - 1) handleSelect(samples[idx+1].filename); }} className="p-1 hover:text-accent transition-colors"><ChevronRight className="w-7 h-7" /></button>
            </div>
            <button onClick={handleSave} className="flex items-center gap-3 px-10 py-3 bg-accent text-black rounded-2xl font-black hover:bg-white transition-all shadow-[0_0_30px_rgba(0,255,65,0.5)] uppercase text-[12px] tracking-[0.1em]">
                <Save className="w-5 h-5" /> Save Changes
            </button>
        </div>
      </header>

      <div className="flex-1 relative flex overflow-hidden">
        <aside className="w-72 border-r border-white/20 bg-[#0a0a0c] flex flex-col z-40 shadow-2xl">
             <div className="p-5 border-b border-white/10 flex items-center gap-3 bg-white/10">
                <Layers className="w-5 h-5 text-accent" />
                <span className="text-[11px] font-cyber text-white font-black uppercase tracking-widest">Active Layers</span>
             </div>
             <div className="flex-1 overflow-y-auto p-3 space-y-2 cyber-scrollbar">
                {currentData.bboxes?.map((b, i) => (
                    <div key={i} onClick={() => setSelectedBBoxIdx(i)} className={`p-3.5 rounded-2xl flex items-center gap-4 cursor-pointer transition-all ${selectedBBoxIdx === i ? 'bg-accent/20 border-2 border-accent/50 shadow-lg scale-[1.02]' : 'bg-white/[0.05] border border-white/10 hover:bg-white/10'}`}>
                        <div className="w-3 h-3 rounded-full shadow-[0_0_10px_currentColor]" style={{ backgroundColor: COLORS[b.category % COLORS.length], color: COLORS[b.category % COLORS.length] }} />
                        <span className={`text-[12px] font-black truncate flex-1 ${selectedBBoxIdx === i ? 'text-white' : 'text-white/80'}`}>{classes[b.category] || `ID_${b.category}`}</span>
                        <Trash2 onClick={(e) => { e.stopPropagation(); const next = [...currentData.bboxes!]; next.splice(i, 1); handleUpdate({ bboxes: next }); if (selectedBBoxIdx === i) setSelectedBBoxIdx(null); }} className="w-4 h-4 text-white/40 hover:text-red-500 transition-colors" />
                    </div>
                ))}
                {currentData.waypoints?.length === 10 && (
                    <div onClick={() => setMode('waypoint')} className={`p-3.5 rounded-2xl flex items-center gap-4 cursor-pointer transition-all ${mode === 'waypoint' ? 'bg-green-500/20 border-2 border-green-500/50 shadow-lg scale-[1.02]' : 'bg-white/[0.05] border border-white/10 hover:bg-white/10'}`}>
                        <div className="w-3 h-3 rounded-full bg-green-500 shadow-[0_0_10px_#22c55e]" />
                        <span className={`text-[12px] flex-1 font-black ${mode === 'waypoint' ? 'text-white' : 'text-white/80'}`}>Unified Path (10 Points)</span>
                        <RotateCcw onClick={(e) => { e.stopPropagation(); handleUpdate({ waypoints: [], control_points: [] }); }} className="w-4 h-4 text-white/40 hover:text-orange-400" />
                    </div>
                )}
             </div>
             <div className="p-8 border-t border-white/10 flex justify-center gap-8 bg-white/5">
                <button onClick={async () => { if (selectedFilename) { await API.labels.duplicate(selectedFilename, `copy_${Date.now()}_${selectedFilename}`); await loadFiles(); } }} title="Duplicate" className="text-white hover:text-accent transition-all hover:scale-125"><Copy className="w-6 h-6" /></button>
                <button onClick={async () => { if (selectedFilename && window.confirm("Reset all labels?")) { await API.labels.reset(selectedFilename); handleSelect(selectedFilename); } }} title="Reset" className="text-white hover:text-orange-400 transition-all hover:scale-125"><RotateCcw className="w-6 h-6" /></button>
                <button onClick={async () => { if (selectedFilename && window.confirm("Delete image?")) { await API.labels.delete(selectedFilename); await loadFiles(); setSelectedFilename(null); } }} title="Delete" className="text-white hover:text-red-500 transition-all hover:scale-125"><Trash2 className="w-6 h-6" /></button>
             </div>
        </aside>

        <main className="flex-1 relative bg-black">
             {selectedFilename ? (
                <Annotator imageSrc={`/api/v1/labels/image/${selectedFilename}`} bboxes={currentData.bboxes || []} waypoints={currentData.waypoints || []} control_points={currentData.control_points || []} mode={mode} onUpdate={handleUpdate} classNames={classes} selectedBBoxIdx={selectedBBoxIdx ?? undefined} onSelectBBox={setSelectedBBoxIdx} />
             ) : (
                <div className="h-full flex flex-col items-center justify-center gap-6 text-white/40 font-cyber tracking-[0.5em] animate-pulse">
                    <div className="w-16 h-16 border-8 border-white/5 border-t-accent rounded-full animate-spin" />
                    SYNCING_DATABASE...
                </div>
             )}
        </main>

        <aside className="w-80 border-l border-white/20 bg-[#0a0a0c] flex flex-col p-7 space-y-12 z-40 overflow-y-auto cyber-scrollbar">
            <section className="space-y-5">
                <h3 className="text-[11px] text-white font-black uppercase tracking-[0.3em] font-cyber border-l-4 border-accent pl-4">Tool Configuration</h3>
                <div className="grid grid-cols-2 gap-4">
                    <button onClick={() => { setMode('bbox'); setSelectedBBoxIdx(null); }} className={`flex flex-col items-center gap-3 p-6 rounded-3xl border-2 transition-all ${mode === 'bbox' ? 'bg-accent/20 border-accent text-white shadow-[0_0_20px_rgba(0,255,65,0.2)]' : 'bg-white/5 border-white/10 text-white/40 hover:bg-white/10'}`}>
                        <BoxIcon className="w-7 h-7" />
                        <span className="text-[11px] font-black tracking-widest uppercase">Objects</span>
                    </button>
                    <button onClick={() => { setMode('waypoint'); setSelectedBBoxIdx(null); }} className={`flex flex-col items-center gap-3 p-6 rounded-3xl border-2 transition-all ${mode === 'waypoint' ? 'bg-accent/20 border-accent text-white shadow-[0_0_20px_rgba(0,255,65,0.2)]' : 'bg-white/5 border-white/10 text-white/40 hover:bg-white/10'}`}>
                        <Move className="w-7 h-7" />
                        <span className="text-[11px] font-black tracking-widest uppercase">Path</span>
                    </button>
                </div>
            </section>

            <section className="space-y-5">
                <h3 className="text-[11px] text-white font-black uppercase tracking-[0.3em] font-cyber border-l-4 border-accent pl-4 flex items-center gap-3"><Zap className="w-4 h-4 text-accent fill-accent" /> Path Templates</h3>
                <div className="grid grid-cols-1 gap-3">
                    <TemplateBtn label="LEFT SWEEP ARC" onClick={() => handleSpawnTemplate('left')} />
                    <TemplateBtn label="DYNAMIC STRAIGHT" onClick={() => handleSpawnTemplate('straight')} />
                    <TemplateBtn label="RIGHT SWEEP ARC" onClick={() => handleSpawnTemplate('right')} />
                </div>
            </section>

            {/* Brighter behavioral select */}
            <section className="space-y-5">
                <h3 className="text-[11px] text-white font-black uppercase tracking-[0.3em] font-cyber border-l-4 border-accent pl-4">Behavioral Logic</h3>
                <div className="relative">
                    <select value={currentData.command} onChange={(e) => setCurrentData(prev => ({ ...prev, command: parseInt(e.target.value) }))} className="w-full bg-[#2a2a2c] border-2 border-white/30 rounded-2xl p-4 text-[13px] text-white font-black outline-none focus:border-accent transition-all appearance-none cursor-pointer shadow-xl">
                        <option value={0}>0: FOLLOW_LANE</option>
                        <option value={1}>1: TURN_LEFT</option>
                        <option value={2}>2: TURN_RIGHT</option>
                        <option value={3}>3: STRAIGHT</option>
                    </select>
                    <div className="absolute right-5 top-1/2 -translate-y-1/2 pointer-events-none text-accent"><ChevronDown className="w-5 h-5" /></div>
                </div>
            </section>

            {/* Smart Search Taxonomy Section */}
            <section className="flex-1 flex flex-col min-h-0 space-y-5">
                <div className="flex justify-between items-center">
                    <h3 className="text-[11px] text-white font-black uppercase tracking-[0.3em] font-cyber border-l-4 border-accent pl-4">Classification</h3>
                    <button onClick={() => setIsClassModalOpen(true)} className="text-white hover:text-accent transition-all hover:rotate-90"><Settings2 className="w-7 h-7" /></button>
                </div>
                
                {/* Search Input */}
                <div className="relative">
                    <input 
                        type="text" 
                        placeholder="Search & Apply Class..." 
                        value={classSearch}
                        onChange={(e) => setClassSearch(e.target.value)}
                        className="w-full bg-white/10 border-2 border-white/20 rounded-2xl p-4 pl-12 text-[13px] text-white font-black placeholder:text-white/20 focus:border-accent outline-none transition-all shadow-inner"
                    />
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/20" />
                </div>

                <div className="flex-1 overflow-y-auto space-y-2 pr-2 cyber-scrollbar max-h-64">
                    {filteredClasses.map((c) => (
                        <button 
                            key={c.id} 
                            onClick={() => {
                                setLastSelectedClass(c.id);
                                setClassSearch('');
                                if (selectedBBoxIdx !== null) {
                                    const next = [...(currentData.bboxes || [])];
                                    next[selectedBBoxIdx] = { ...next[selectedBBoxIdx], category: c.id };
                                    handleUpdate({ bboxes: next });
                                }
                            }}
                            className={`w-full flex items-center gap-4 p-4 rounded-2xl border-2 transition-all group ${lastSelectedClass === c.id ? 'bg-accent text-black border-accent' : 'bg-white/[0.06] border-white/5 hover:bg-white/10'}`}
                        >
                            <div className={`w-3.5 h-3.5 rounded-full shadow-[0_0_8px_currentColor] ${lastSelectedClass === c.id ? 'bg-black text-black' : ''}`} style={{ backgroundColor: lastSelectedClass === c.id ? undefined : COLORS[c.id % COLORS.length], color: COLORS[c.id % COLORS.length] }} />
                            <span className="text-[13px] font-black truncate text-left flex-1 uppercase tracking-wider">{c.id}: {c.name}</span>
                            {lastSelectedClass === c.id && <Zap className="w-4 h-4 fill-black" />}
                        </button>
                    ))}
                </div>
            </section>
        </aside>
      </div>

      {isClassModalOpen && (
        <div className="fixed inset-0 bg-black/95 backdrop-blur-xl z-[100] flex items-center justify-center p-6">
            <div className="w-full max-w-2xl bg-[#0a0a0c] border-2 border-white/20 rounded-[3rem] overflow-hidden shadow-[0_0_150px_rgba(0,0,0,1)] animate-in zoom-in-95 duration-300">
                <div className="p-10 border-b border-white/10 flex justify-between items-center bg-white/5">
                    <h2 className="text-3xl font-black text-white flex items-center gap-5 font-cyber tracking-tighter"><Settings2 className="w-10 h-10 text-accent" /> SCHEMA_CONFIG</h2>
                    <button onClick={() => setIsClassModalOpen(false)} className="p-4 hover:bg-white/10 rounded-full transition-colors text-white hover:text-white"><X className="w-10 h-10" /></button>
                </div>
                <div className="p-12 space-y-6 max-h-[600px] overflow-y-auto cyber-scrollbar">
                    {classes.map((c, i) => (
                        <div key={i} className="flex gap-5 group">
                            <div className="w-20 flex items-center justify-center font-black text-lg text-accent bg-accent/10 rounded-3xl border-2 border-accent/20">#{i}</div>
                            <input value={c} onChange={(e) => { const next = [...classes]; next[i] = e.target.value; setClasses(next); }} className="flex-1 bg-black border-2 border-white/20 rounded-3xl p-5 text-[18px] text-white font-black focus:border-accent outline-none transition-all shadow-inner" />
                            <button onClick={() => { const next = classes.filter((_, idx) => idx !== i); setClasses(next); }} className="p-5 hover:bg-red-500/20 text-white hover:text-red-500 rounded-3xl transition-all border border-transparent hover:border-red-500/40"><X className="w-8 h-8" /></button>
                        </div>
                    ))}
                    <button onClick={() => setClasses([...classes, "New Class"])} className="w-full p-6 rounded-[2rem] border-4 border-dashed border-white/10 text-white/60 hover:text-accent hover:border-accent/40 transition-all text-sm font-black uppercase tracking-[0.3em]">+ ATTACH_VIRTUAL_ID</button>
                </div>
                <div className="p-12 bg-white/[0.05] border-t border-white/10 flex gap-8">
                    <button onClick={() => setIsClassModalOpen(false)} className="flex-1 py-6 rounded-3xl border-2 border-white/20 text-white text-lg font-black hover:bg-white/5 transition-all uppercase tracking-widest">Abort</button>
                    <button onClick={async () => { await API.labels.updateClasses(projectId, classes); setIsClassModalOpen(false); }} className="flex-1 py-6 rounded-3xl bg-accent text-black text-lg font-black shadow-[0_0_50px_rgba(0,255,65,0.5)] hover:scale-[1.03] transition-all uppercase tracking-widest">Commit_Schema</button>
                </div>
            </div>
        </div>
      )}
    </div>
  );
};

const TemplateBtn = ({ label, onClick }: { label: string, onClick: () => void }) => (
    <button onClick={onClick} className="p-5 bg-white/[0.03] hover:bg-white/10 rounded-3xl text-[12px] font-black border border-white/10 hover:border-accent transition-all flex justify-between items-center text-white group shadow-2xl">
        <span className="group-hover:text-accent transition-colors tracking-widest uppercase">{label}</span>
        <ChevronRight className="w-5 h-5 opacity-40 group-hover:opacity-100 group-hover:translate-x-2 transition-all" />
    </button>
);
