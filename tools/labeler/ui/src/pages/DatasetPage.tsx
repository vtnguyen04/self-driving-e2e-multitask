import React, { useCallback, useEffect, useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { API } from '../api';
import { MainLayout } from '../components/layout/MainLayout';
import { ArrowUpRight, BarChart3, CheckCircle2, Search, ChevronDown } from 'lucide-react';
import { Sample } from '../types';

const COLORS = [
  '#ff003c', '#00ff41', '#00f3ff', '#ffcc00', '#ff00ff',
  '#00ffff', '#ffff00', '#ff8800', '#88ff00', '#00ff88',
  '#0088ff', '#8800ff', '#ff0088', '#ffffff'
];

const getBezierPoint = (p0: any, p1: any, p2: any, p3: any, t: number) => {
    const cx = (1 - t) ** 3 * p0.x + 3 * (1 - t) ** 2 * t * p1.x + 3 * (1 - t) * t ** 2 * p2.x + t ** 3 * p3.x;
    const cy = (1 - t) ** 3 * p0.y + 3 * (1 - t) ** 2 * t * p1.y + 3 * (1 - t) * t ** 2 * p2.y + t ** 3 * p3.y;
    return { x: cx, y: cy };
};

const MiniPreview: React.FC<{ sample: Sample }> = ({ sample }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d')!;
        const img = new Image();
        img.src = `/api/v1/labels/image/${sample.filename}`;
        img.onload = () => {
            canvas.width = 240;
            canvas.height = 240;
            ctx.drawImage(img, 0, 0, 240, 240);
            
            let data: any = null;
            try {
                if (typeof sample.data === 'string') data = JSON.parse(sample.data);
                else if (typeof sample.data === 'object') data = sample.data;
                else if (sample.bboxes) data = { bboxes: sample.bboxes, waypoints: sample.waypoints };
            } catch(e) {}

            if (data) {
                // Draw BBoxes
                data.bboxes?.forEach((b: any) => {
                    const color = COLORS[b.category % COLORS.length];
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 3;
                    ctx.strokeRect((b.cx - b.w/2) * 240, (b.cy - b.h/2) * 240, b.w * 240, b.h * 240);
                });
                
                // Draw Trajectory (Bezier or Multi-point)
                if (data.waypoints?.length >= 2) {
                    ctx.strokeStyle = '#00ff41';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    
                    if (data.waypoints.length === 4) {
                        const [p0, p1, p2, p3] = data.waypoints;
                        for (let t = 0; t <= 1; t += 0.1) {
                            const pt = getBezierPoint(p0, p1, p2, p3, t);
                            if (t === 0) ctx.moveTo(pt.x * 240, pt.y * 240);
                            else ctx.lineTo(pt.x * 240, pt.y * 240);
                        }
                    } else {
                        data.waypoints.forEach((p: any, i: number) => {
                            if (i === 0) ctx.moveTo(p.x * 240, p.y * 240);
                            else ctx.lineTo(p.x * 240, p.y * 240);
                        });
                    }
                    ctx.stroke();

                    // Draw points
                    data.waypoints.forEach((p: any) => {
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

  const [stats, setStats] = useState<any>(null);
  const [classes, setClasses] = useState<string[]>([]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [filterClass, setFilterClass] = useState<number | null>(null);
  const [filterStatus, setFilterStatus] = useState<'all' | 'labeled' | 'unlabeled'>('all');
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const LIMIT = 48;

  const loadData = useCallback(async (isLoadMore = false) => {
    const s = await API.labels.getStats(projectId);
    setStats(s);
    const c = await API.labels.getClasses(projectId);
    setClasses(c || []);
    
    const currentOffset = isLoadMore ? offset + LIMIT : 0;
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
    setOffset(currentOffset);
    setHasMore(list.length === LIMIT);
  }, [projectId, filterStatus, filterClass, offset]);

  useEffect(() => { setOffset(0); loadData(false); }, [filterStatus, filterClass]);

  if (!stats) return <div className="h-screen flex items-center justify-center text-accent font-cyber animate-pulse tracking-[0.5em]">SYSTEM_INITIALIZING...</div>;

  const progress = (stats.labeled / stats.total) * 100 || 0;

  return (
    <MainLayout>
      <div className="p-8 space-y-8 h-full overflow-y-auto cyber-scrollbar bg-[#050505]">
        
        <div className="flex justify-between items-end border-b border-white/5 pb-6">
            <div>
                <h1 className="text-2xl font-bold text-white mb-2 flex items-center gap-3 font-cyber tracking-tight">
                    <BarChart3 className="w-6 h-6 text-accent" />
                    DASHBOARD_V2
                </h1>
                <p className="text-white/40 text-[10px] font-mono uppercase tracking-[0.3em]">Project Integrity: High-Performance</p>
            </div>
            <button onClick={() => navigate(`/annotate/${projectId}`)} className="px-8 py-3 bg-accent text-black font-bold rounded-2xl hover:scale-105 transition-all shadow-[0_0_30px_rgba(0,255,65,0.2)] flex items-center gap-2 uppercase text-xs">
                Annotate New Data <ArrowUpRight className="w-4 h-4" />
            </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-6 rounded-3xl bg-[#0a0a0c] border border-white/5 relative overflow-hidden">
                <h3 className="text-white/30 text-[9px] uppercase tracking-[0.2em] mb-4 font-cyber">Overall Progress</h3>
                <div className="text-4xl font-bold text-white mb-2">{stats.labeled} <span className="text-lg text-white/20">/ {stats.total}</span></div>
                <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <div className="h-full bg-accent" style={{ width: `${progress}%` }} />
                </div>
                <p className="mt-3 text-accent text-[10px] font-bold">{progress.toFixed(1)}% COMPLETE</p>
            </div>

            <div className="p-6 rounded-3xl bg-[#0a0a0c] border border-white/5">
                <h3 className="text-white/30 text-[9px] uppercase tracking-[0.2em] mb-4 font-cyber">Backlog</h3>
                <div className="text-4xl font-bold text-orange-500 mb-2">{stats.total - stats.labeled}</div>
                <p className="text-white/40 text-[10px]">Samples requiring attention</p>
            </div>

            <div className="p-6 rounded-3xl bg-[#0a0a0c] border border-white/5 flex flex-col">
                <h3 className="text-white/30 text-[9px] uppercase tracking-[0.2em] mb-4 font-cyber">Class Balance</h3>
                <div className="flex-1 overflow-y-auto space-y-2 pr-2 cyber-scrollbar max-h-32">
                    {classes.map((c, i) => (
                        <div key={i} className="flex items-center gap-3 text-[10px]">
                            <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                            <span className="flex-1 text-white/60 truncate">{c}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>

        <div className="space-y-6 pt-4">
            <div className="flex justify-between items-center bg-[#0a0a0c]/50 p-4 rounded-2xl border border-white/5 sticky top-0 z-10 backdrop-blur-xl">
                <h2 className="text-sm font-bold text-white flex items-center gap-3 font-cyber tracking-widest uppercase">
                    <Search className="w-4 h-4 text-accent" /> Resource Browser
                </h2>
                <div className="flex gap-3">
                    <select className="bg-black border border-white/10 text-white text-[10px] font-bold rounded-xl px-4 py-2 outline-none" value={filterStatus} onChange={(e) => setFilterStatus(e.target.value as any)}>
                        <option value="all">ALL DATA</option>
                        <option value="unlabeled">TODO</option>
                        <option value="labeled">COMPLETED</option>
                    </select>
                    <select className="bg-black border border-white/10 text-white text-[10px] font-bold rounded-xl px-4 py-2 outline-none" value={filterClass || ''} onChange={(e) => setFilterClass(e.target.value ? parseInt(e.target.value) : null)}>
                        <option value="">ALL CLASSES</option>
                        {classes.map((c, i) => <option key={i} value={i}>{c}</option>)}
                    </select>
                </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {samples.map((s) => (
                    <div key={s.filename} onClick={() => navigate(`/annotate/${projectId}?file=${s.filename}`)} className="aspect-square bg-[#0a0a0c] border border-white/5 rounded-2xl overflow-hidden relative group cursor-pointer hover:border-accent/50 transition-all shadow-2xl">
                        <MiniPreview sample={s} />
                        {s.is_labeled && (
                            <div className="absolute top-3 right-3 bg-accent text-black p-1 rounded-lg shadow-lg">
                                <CheckCircle2 className="w-3.5 h-3.5" />
                            </div>
                        )}
                        <div className="absolute bottom-0 left-0 w-full p-2.5 bg-black/80 backdrop-blur-md border-t border-white/5">
                            <p className="text-[8px] text-white/40 font-mono truncate">{s.filename}</p>
                        </div>
                    </div>
                ))}
            </div>

            {hasMore && (
                <div className="flex justify-center py-10">
                    <button onClick={() => loadData(true)} className="px-10 py-3 rounded-2xl border border-white/10 hover:border-accent text-white/40 hover:text-accent font-cyber text-[10px] tracking-widest transition-all flex items-center gap-2 uppercase">
                        Load More Resources <ChevronDown className="w-4 h-4" />
                    </button>
                </div>
            )}
        </div>
      </div>
    </MainLayout>
  );
};
