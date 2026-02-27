import { ArrowUpRight, BarChart3, CheckCircle2, Search, Trash2, Upload } from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { API } from '../api';
import { MainLayout } from '../components/layout/MainLayout';
import { UploadModal } from '../components/UploadModal';
import { BBox, Sample, Stats, Waypoint } from '../types';

const COLORS = [
  '#ff003c', '#00ff41', '#00f3ff', '#ffcc00', '#ff00ff',
  '#00ffff', '#ffff00', '#ff8800', '#88ff00', '#00ff88',
  '#0088ff', '#8800ff', '#ff0088', '#ffffff'
];

interface Point {
    x: number;
    y: number;
}

const getBezierPoint = (p0: Point, p1: Point, p2: Point, p3: Point, t: number): Point => {
    const cx = (1 - t) ** 3 * p0.x + 3 * (1 - t) ** 2 * t * p1.x + 3 * (1 - t) * t ** 2 * p2.x + t ** 3 * p3.x;
    const cy = (1 - t) ** 3 * p0.y + 3 * (1 - t) ** 2 * t * p1.y + 3 * (1 - t) * t ** 2 * p2.y + t ** 3 * p3.y;
    return { x: cx, y: cy };
};

const MiniPreview: React.FC<{ sample: Sample }> = ({ sample }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const img = new Image();
        img.src = `/api/v1/labels/image/${sample.filename}`;
        img.onload = () => {
            canvas.width = 240;
            canvas.height = 240;
            ctx.drawImage(img, 0, 0, 240, 240);

            interface LabelData {
                bboxes?: BBox[];
                waypoints?: Waypoint[];
            }

            let data: LabelData | null = null;
            try {
                if (typeof sample.data === 'string') {
                    data = JSON.parse(sample.data) as LabelData;
                } else if (typeof sample.data === 'object' && sample.data !== null) {
                    data = sample.data as unknown as LabelData;
                } else if (sample.bboxes) {
                    data = { bboxes: sample.bboxes, waypoints: sample.waypoints };
                }
            } catch(error) {
                console.error("Failed to parse sample data", error);
            }

            if (data) {
                // Draw BBoxes
                data.bboxes?.forEach((b) => {
                    const color = COLORS[b.category % COLORS.length];
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 3;
                    ctx.strokeRect((b.cx - b.w/2) * 240, (b.cy - b.h/2) * 240, b.w * 240, b.h * 240);
                });

                // Draw Trajectory (Bezier or Multi-point)
                const waypoints = data.waypoints || [];
                if (waypoints.length >= 2) {
                    ctx.strokeStyle = '#00ff41';
                    ctx.lineWidth = 3;
                    ctx.beginPath();

                    if (waypoints.length === 4) {
                        const [p0, p1, p2, p3] = waypoints;
                        for (let t = 0; t <= 1; t += 0.1) {
                            const pt = getBezierPoint(p0, p1, p2, p3, t);
                            if (t === 0) ctx.moveTo(pt.x * 240, pt.y * 240);
                            else ctx.lineTo(pt.x * 240, pt.y * 240);
                        }
                    } else {
                        waypoints.forEach((p, i) => {
                            if (i === 0) ctx.moveTo(p.x * 240, p.y * 240);
                            else ctx.lineTo(p.x * 240, p.y * 240);
                        });
                    }
                    ctx.stroke();

                    // Draw points
                    waypoints.forEach((p) => {
                        ctx.fillStyle = '#00ff41';
                        ctx.beginPath();
                        ctx.arc(p.x * 240, p.y * 240, 3, 0, Math.PI * 2);
                        ctx.fill();
                    });
                }
            }
        };
    }, [sample]);

    return <canvas ref={canvasRef} className="w-full h-full object-cover opacity-70 group-hover:opacity-100 transition-all duration-500" />;
};

export const DatasetPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const projectId = id ? parseInt(id) : 1;
  const navigate = useNavigate();

  const [stats, setStats] = useState<Stats | null>(null);
  const [classes, setClasses] = useState<string[]>([]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [filterClass, setFilterClass] = useState<number | null>(null);
  const [filterStatus, setFilterStatus] = useState<'all' | 'labeled' | 'unlabeled'>('all');
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [bulkMode, setBulkMode] = useState(false);
  const dragState = useRef({ isDragging: false, action: null as 'add' | 'remove' | null });

  const LIMIT = 48;

  const loadData = useCallback(async (isLoadMore = false) => {
    if (loading) return;
    setLoading(true);
    try {
        const [s, c] = await Promise.all([
            API.labels.getStats(projectId),
            API.labels.getClasses(projectId)
        ]);
        setStats(s);
        setClasses(c || []);

        const currentOffset = isLoadMore ? samples.length : 0;
        const list = await API.labels.list({
            limit: LIMIT,
            offset: currentOffset,
            project_id: projectId,
            is_labeled: filterStatus === 'all' ? undefined : (filterStatus === 'labeled'),
            class_id: filterClass ?? undefined
        });

        if (isLoadMore) {
            setSamples(prev => [...prev, ...list]);
        } else {
            setSamples(list);
        }
        setHasMore(list.length === LIMIT);
    } catch (err) {
        console.error("Load failed", err);
    } finally {
        setLoading(false);
    }
  }, [projectId, filterStatus, filterClass, samples.length, loading]);

  useEffect(() => {
    loadData(false).catch(console.error);
  }, [projectId, filterStatus, filterClass]);

  if (!stats) return <div className="h-screen flex items-center justify-center text-accent font-cyber animate-pulse tracking-[0.5em]">SYSTEM_INITIALIZING...</div>;

  const progress = (stats.labeled / stats.total) * 100 || 0;

  return (
    <MainLayout>
      <div className="p-8 space-y-8 h-full overflow-y-auto cyber-scrollbar bg-[#050505]">

        <div className="flex justify-between items-end border-b border-white/5 pb-6">
            <div>
                <h1 className="text-4xl font-black text-white mb-3 flex items-center gap-4 font-cyber tracking-tight">
                    <BarChart3 className="w-8 h-8 text-accent" />
                    DASHBOARD_V2
                </h1>
                <p className="text-white text-base font-bold font-mono uppercase tracking-[0.15em]">Project Integrity: High-Performance</p>
            </div>
            <div className="flex gap-4">
                <button
                    onClick={async () => {
                        if (!confirm(`⚠️ DELETE PROJECT?\n\nThis will permanently delete ALL samples and data.\n\nType 'DELETE' to confirm.`)) return;
                        const userInput = prompt('Type DELETE to confirm:');
                        if (userInput !== 'DELETE') {
                            alert('Deletion cancelled');
                            return;
                        }
                        try {
                            await fetch(`/api/v1/labels/projects/${projectId}`, { method: 'DELETE' });
                            navigate('/');
                        } catch (error) {
                            console.error('Delete project failed:', error);
                            alert('Failed to delete project');
                        }
                    }}
                    className="px-6 py-3 bg-red-500/10 text-red-400 font-bold rounded-2xl hover:scale-105 transition-all border-2 border-red-500/30 hover:border-red-500/60 flex items-center gap-2 uppercase text-xs"
                >
                    <Trash2 className="w-4 h-4" /> Delete Project
                </button>
                <button onClick={() => setShowUploadModal(true)} className="px-8 py-3 bg-white/10 text-white font-bold rounded-2xl hover:scale-105 transition-all border-2 border-white/20 flex items-center gap-2 uppercase text-xs">
                    <Upload className="w-4 h-4" /> Upload Data
                </button>
                <button onClick={() => navigate(`/annotate/${projectId}`)} className="px-8 py-3 bg-accent text-black font-bold rounded-2xl hover:scale-105 transition-all shadow-[0_0_30px_rgba(0,255,65,0.2)] flex items-center gap-2 uppercase text-xs">
                    Annotate New Data <ArrowUpRight className="w-4 h-4" />
                </button>
            </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-6 rounded-3xl bg-[#0a0a0c] border border-white/5 relative overflow-hidden">
                <h3 className="text-white text-sm font-black uppercase tracking-[0.1em] mb-4 font-cyber">Overall Progress</h3>
                <div className="text-6xl font-black text-white mb-3">{stats.labeled} <span className="text-3xl text-white/80 font-black">/ {stats.total}</span></div>
                <div className="w-full h-3 bg-white/20 rounded-full overflow-hidden">
                    <div className="h-full bg-accent shadow-[0_0_20px_rgba(0,255,65,0.5)]" style={{ width: `${progress}%` }} />
                </div>
                <p className="mt-4 text-accent text-base font-black">{progress.toFixed(1)}% COMPLETE</p>
            </div>

            <div className="p-6 rounded-3xl bg-[#0a0a0c] border border-white/5">
                <h3 className="text-white text-sm font-black uppercase tracking-[0.1em] mb-4 font-cyber">Backlog</h3>
                <div className="text-6xl font-black text-orange-400 mb-3">{stats.total - stats.labeled}</div>
                <p className="text-white text-base font-bold">Samples requiring attention</p>
            </div>

            <div className="p-6 rounded-3xl bg-[#0a0a0c] border border-white/5 flex flex-col">
                <h3 className="text-white text-sm font-black uppercase tracking-[0.1em] mb-4 font-cyber">Class Balance</h3>
                <div className="flex-1 overflow-y-auto space-y-3 pr-2 cyber-scrollbar max-h-32">
                    {classes.map((c, i) => (
                        <div key={i} className="flex items-center gap-3 text-base">
                            <div className="w-3 h-3 rounded-full shadow-lg" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                            <span className="flex-1 text-white font-bold truncate">{c}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>

        <div className="space-y-6 pt-4">
            <div className="flex justify-between items-center bg-[#0a0a0c]/50 p-4 rounded-2xl border border-white/5 sticky top-0 z-10 backdrop-blur-xl">
                <h2 className="text-xl font-black text-white flex items-center gap-4 font-cyber tracking-wide uppercase">
                    <Search className="w-6 h-6 text-accent" /> Resource Browser
                </h2>
                <div className="flex gap-3">
                    {bulkMode && selectedFiles.size > 0 && (
                        <button
                            onClick={async () => {
                                if (!confirm(`Delete ${selectedFiles.size} selected files?`)) return;
                                try {
                                    await fetch('/api/v1/labels/batch/delete', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ filenames: Array.from(selectedFiles) })
                                    });
                                    setSelectedFiles(new Set());
                                    setBulkMode(false);
                                    loadData(false);
                                } catch (error) {
                                    console.error('Bulk delete failed:', error);
                                }
                            }}
                            className="px-6 py-2 bg-red-500/20 text-red-400 border-2 border-red-500/40 font-bold rounded-xl hover:bg-red-500/30 transition-all flex items-center gap-2 text-sm uppercase"
                        >
                            <Trash2 className="w-4 h-4" /> Delete ({selectedFiles.size})
                        </button>
                    )}
                    <button
                        onClick={() => {
                            setBulkMode(!bulkMode);
                            setSelectedFiles(new Set());
                        }}
                        className={`px-6 py-2 font-bold rounded-xl transition-all text-sm uppercase ${
                            bulkMode
                                ? 'bg-accent/20 text-accent border-2 border-accent'
                                : 'bg-white/10 text-white border-2 border-white/20 hover:border-white/40'
                        }`}
                    >
                        {bulkMode ? 'Cancel Selection' : 'Select Multiple'}
                    </button>
                    {bulkMode && (
                        <button
                            onClick={() => {
                                const allCurrentLocally = new Set(samples.map(s => s.filename));
                                setSelectedFiles(allCurrentLocally);
                            }}
                            className="px-6 py-2 bg-white/10 text-white border-2 border-white/20 hover:border-white/40 font-bold rounded-xl transition-all text-sm uppercase"
                        >
                            Select All (View)
                        </button>
                    )}
                    <select
                        className="bg-[#1a1a1c] border-2 border-white/30 text-white text-base font-black rounded-xl px-6 py-3 outline-none hover:border-accent transition-colors"
                        value={filterStatus}
                        onChange={(e) => setFilterStatus(e.target.value as 'all' | 'labeled' | 'unlabeled')}
                    >
                        <option value="all">ALL DATA</option>
                        <option value="unlabeled">TODO</option>
                        <option value="labeled">COMPLETED</option>
                    </select>
                    <select
                        className="bg-[#1a1a1c] border-2 border-white/30 text-white text-base font-black rounded-xl px-6 py-3 outline-none hover:border-accent transition-colors"
                        value={filterClass || ''}
                        onChange={(e) => setFilterClass(e.target.value ? parseInt(e.target.value) : null)}
                    >
                        <option value="">ALL CLASSES</option>
                        {classes.map((c, i) => <option key={i} value={i}>{c}</option>)}
                    </select>
                </div>
            </div>


            <div
                className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4"
                onPointerLeave={() => { dragState.current.isDragging = false; dragState.current.action = null; }}
                onPointerUp={() => { dragState.current.isDragging = false; dragState.current.action = null; }}
                style={{ touchAction: bulkMode ? 'none' : 'auto' }}
            >
                {samples.map((s) => (
                    <div
                        key={s.filename}
                        onClick={(e) => {
                            if (bulkMode) {
                                e.stopPropagation();
                            } else {
                                navigate(`/annotate/${projectId}?file=${s.filename}`);
                            }
                        }}
                        onPointerDown={(e) => {
                            if (!bulkMode) return;
                            e.currentTarget.releasePointerCapture(e.pointerId);
                            dragState.current.isDragging = true;
                            const action = selectedFiles.has(s.filename) ? 'remove' : 'add';
                            dragState.current.action = action;
                            const newSelected = new Set(selectedFiles);
                            if (action === 'add') newSelected.add(s.filename); else newSelected.delete(s.filename);
                            setSelectedFiles(newSelected);
                        }}
                        onPointerEnter={() => {
                            if (!bulkMode || !dragState.current.isDragging || !dragState.current.action) return;
                            const newSelected = new Set(selectedFiles);
                            if (dragState.current.action === 'add') newSelected.add(s.filename); else newSelected.delete(s.filename);
                            setSelectedFiles(newSelected);
                        }}
                        className={`aspect-square bg-[#0a0a0c] border rounded-2xl overflow-hidden relative group cursor-pointer transition-all shadow-2xl ${
                            bulkMode && selectedFiles.has(s.filename)
                                ? 'border-accent border-4 scale-95'
                                : 'border-white/5 hover:border-accent/50'
                        }`}
                    >
                        {bulkMode && (
                            <div className="absolute top-3 left-3 z-10">
                                <input
                                    type="checkbox"
                                    checked={selectedFiles.has(s.filename)}
                                    onChange={() => {}}
                                    className="w-6 h-6 accent-accent cursor-pointer"
                                />
                            </div>
                        )}
                        <MiniPreview sample={s} />
                        {s.is_labeled && (
                            <div className="absolute top-3 right-3 bg-accent text-black p-1 rounded-lg shadow-lg">
                                <CheckCircle2 className="w-3.5 h-3.5" />
                            </div>
                        )}
                        <div className="absolute bottom-0 left-0 w-full p-3 bg-black/95 backdrop-blur-md border-t border-white/20">
                            <p className="text-sm text-white font-mono font-bold truncate">{s.filename}</p>
                        </div>
                    </div>
                ))}
            </div>

            {hasMore && (
            <div className="mt-12 flex justify-center pb-20">
                <button
                  onClick={() => loadData(true)}
                  disabled={loading}
                  className="group relative px-12 py-4 bg-[#0a0a0c] border-2 border-accent/20 hover:border-accent text-accent overflow-hidden transition-all active:scale-95 disabled:opacity-50 disabled:cursor-wait"
                >
                  <div className="relative z-10 flex items-center gap-3">
                    {loading ? (
                        <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                    ) : (
                        <div className="w-2 h-2 bg-accent rounded-full group-hover:animate-ping" />
                    )}
                    <span className="font-cyber font-black text-lg tracking-[0.3em] uppercase transition-all group-hover:tracking-[0.5em]">
                        {loading ? 'CALCULATING_VECTORS...' : 'LOAD_MORE_RESOURCES'}
                    </span>
                  </div>
                  <div className="absolute inset-0 bg-accent/5 translate-y-full group-hover:translate-y-0 transition-transform" />
                </button>
            </div>
          )}
        </div>
      </div>

      {showUploadModal && (
        <UploadModal
          projectId={projectId}
          onClose={() => setShowUploadModal(false)}
          onSuccess={() => {
            loadData(false);
          }}
        />
      )}
    </MainLayout>
  );
};
