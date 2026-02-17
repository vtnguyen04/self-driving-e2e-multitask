import { Check, Search } from 'lucide-react';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useCanvasTransform } from '../../hooks/useCanvasTransform';
import { BBox, Waypoint } from '../../types';

interface AnnotatorProps {
  imageSrc: string;
  bboxes: BBox[];
  waypoints: Waypoint[];
  control_points?: Waypoint[];
  mode: 'bbox' | 'waypoint';
  onUpdate: (data: { bboxes?: BBox[]; waypoints?: Waypoint[]; control_points?: Waypoint[] }) => void;
  selectedBBoxIdx?: number;
  onSelectBBox?: (idx: number | null) => void;
  classNames?: string[];
}

const COLORS = [
  '#ff003c', '#00ff41', '#00f3ff', '#ffcc00', '#ff00ff',
  '#00ffff', '#ffff00', '#ff8800', '#88ff00', '#00ff88',
  '#0088ff', '#8800ff', '#ff0088', '#ffffff'
];

const THEME = {
  accent: '#00ff41',
  bboxSelected: '#ffffff',
  waypoint: '#00ff41',
  handle: '#ffcc00',
  crosshair: 'rgba(255, 255, 255, 0.6)',
  font: 'bold 14px Orbitron, sans-serif'
};

const getBezierPoint = (p0: Waypoint, p1: Waypoint, p2: Waypoint, p3: Waypoint, t: number) => {
  const cx = (1 - t) ** 3 * p0.x + 3 * (1 - t) ** 2 * t * p1.x + 3 * (1 - t) * t ** 2 * p2.x + t ** 3 * p3.x;
  const cy = (1 - t) ** 3 * p0.y + 3 * (1 - t) ** 2 * t * p1.y + 3 * (1 - t) * t ** 2 * p2.y + t ** 3 * p3.y;
  return { x: cx, y: cy };
};

export const Annotator: React.FC<AnnotatorProps> = ({
  imageSrc, bboxes, waypoints, control_points = [], mode, onUpdate, selectedBBoxIdx, onSelectBBox, classNames = []
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const quickSearchRef = useRef<HTMLInputElement>(null);

  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const { transform, setTransform, handleZoom, handlePan } = useCanvasTransform();

  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(null);
  const [dragInfo, setDragInfo] = useState<{ type: string; index: number; handle?: string; startX: number; startY: number } | null>(null);
  const [mouseWorld, setMouseWorld] = useState({ x: 0, y: 0 });

  // Floating Search State
  const [searchTerm, setSearchSearchTerm] = useState('');

  useEffect(() => {
    const image = new Image();
    image.src = imageSrc;
    image.onload = () => {
      setImg(image);
      setTransform({ scale: 1, x: 0, y: 0 });
    };
  }, [imageSrc, setTransform]);

  useEffect(() => {
    if (selectedBBoxIdx !== undefined && selectedBBoxIdx !== null) {
        setTimeout(() => quickSearchRef.current?.focus(), 50);
        setSearchSearchTerm('');
    }
  }, [selectedBBoxIdx]);

  const toWorld = useCallback((clientX: number, clientY: number) => {
    if (!canvasRef.current || !img) return { x: 0, y: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    const lx = (clientX - rect.left - transform.x) / transform.scale;
    const ly = (clientY - rect.top - transform.y) / transform.scale;
    return { x: lx / canvasRef.current.width, y: ly / canvasRef.current.height };
  }, [transform, img]);

  const fromWorld = useCallback((worldX: number, worldY: number) => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const vx = (worldX * canvasRef.current.width) * transform.scale + transform.x;
    const vy = (worldY * canvasRef.current.height) * transform.scale + transform.y;
    return { x: vx, y: vy };
  }, [transform]);

  const draw = useCallback(() => {
    if (!img || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    const container = canvas.parentElement!;

    const ratio = img.width / img.height;
    canvas.width = container.clientWidth;
    canvas.height = container.clientWidth / ratio;
    if (canvas.height > container.clientHeight) {
        canvas.height = container.clientHeight;
        canvas.width = container.clientHeight * ratio;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.scale, transform.scale);

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Crosshairs
    ctx.strokeStyle = THEME.crosshair;
    ctx.lineWidth = 1.5 / transform.scale;
    ctx.beginPath();
    ctx.moveTo(mouseWorld.x * canvas.width, 0); ctx.lineTo(mouseWorld.x * canvas.width, canvas.height);
    ctx.moveTo(0, mouseWorld.y * canvas.height); ctx.lineTo(canvas.width, mouseWorld.y * canvas.height);
    ctx.stroke();

    // BBoxes
    bboxes.forEach((box, idx) => {
      const isSelected = selectedBBoxIdx === idx;
      const color = COLORS[box.category % COLORS.length];
      const x = (box.cx - box.w/2) * canvas.width;
      const y = (box.cy - box.h/2) * canvas.height;
      const w = box.w * canvas.width;
      const h = box.h * canvas.height;

      ctx.strokeStyle = isSelected ? THEME.bboxSelected : color;
      ctx.lineWidth = (isSelected ? 4 : 2.5) / transform.scale;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = isSelected ? 'rgba(255,255,255,0.2)' : `${color}33`;
      ctx.fillRect(x, y, w, h);

      if (isSelected) {
          const hs = 6 / transform.scale;
          ctx.fillStyle = THEME.bboxSelected;
          [[x,y], [x+w,y], [x,y+h], [x+w,y+h], [x+w/2,y], [x+w/2,y+h], [x,y+h/2], [x+w,y+h/2]].forEach(([px, py]) => {
              ctx.fillRect(px - hs, py - hs, hs*2, hs*2);
          });
      }

      const label = classNames[box.category] || `CLASS_${box.category}`;
      ctx.font = `bold ${14 / transform.scale}px Orbitron, sans-serif`;
      const textW = ctx.measureText(label).width;
      ctx.fillStyle = isSelected ? THEME.bboxSelected : color;
      ctx.fillRect(x - (1/transform.scale), y - (24/transform.scale), textW + (12/transform.scale), (24/transform.scale));
      ctx.fillStyle = '#000';
      ctx.fillText(label, x + (5/transform.scale), y - (7/transform.scale));
    });

    // Trajectory
    if (waypoints.length >= 2) {
        ctx.strokeStyle = THEME.waypoint;
        ctx.lineWidth = 5 / transform.scale;
        ctx.beginPath();
        waypoints.forEach((p, i) => {
            if (i === 0) ctx.moveTo(p.x * canvas.width, p.y * canvas.height);
            else ctx.lineTo(p.x * canvas.width, p.y * canvas.height);
        });
        ctx.stroke();

        if (control_points.length === 4) {
            const [p0, p1, p2, p3] = control_points;
            ctx.strokeStyle = THEME.handle;
            ctx.setLineDash([5 / transform.scale, 5 / transform.scale]);
            ctx.beginPath();
            ctx.moveTo(p0.x * canvas.width, p0.y * canvas.height); ctx.lineTo(p1.x * canvas.width, p1.y * canvas.height);
            ctx.moveTo(p3.x * canvas.width, p3.y * canvas.height); ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
            ctx.stroke();
            ctx.setLineDash([]);

            control_points.forEach((p, i) => {
                const isAnchor = i === 0 || i === 3;
                ctx.fillStyle = isAnchor ? THEME.waypoint : THEME.handle;
                ctx.beginPath();
                ctx.arc(p.x * canvas.width, p.y * canvas.height, (isAnchor ? 8 : 6) / transform.scale, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2 / transform.scale;
                ctx.stroke();
            });
        }

        waypoints.forEach((p) => {
            ctx.fillStyle = '#fff';
            ctx.beginPath();
            ctx.arc(p.x * canvas.width, p.y * canvas.height, 3 / transform.scale, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    if (isDrawing && startPos) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3 / transform.scale;
        ctx.setLineDash([8 / transform.scale, 8 / transform.scale]);
        const x = Math.min(startPos.x, mouseWorld.x) * canvas.width;
        const y = Math.min(startPos.y, mouseWorld.y) * canvas.height;
        const w = Math.abs(mouseWorld.x - startPos.x) * canvas.width;
        const h = Math.abs(mouseWorld.y - startPos.y) * canvas.height;
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
    }

    ctx.restore();
  }, [img, bboxes, waypoints, control_points, transform, selectedBBoxIdx, isDrawing, startPos, mouseWorld, classNames]);

  useEffect(() => { draw(); }, [draw]);

  const onMouseDown = (e: React.MouseEvent) => {
    const world = toWorld(e.clientX, e.clientY);
    if (e.altKey || e.button === 1) {
        setDragInfo({ type: 'pan', index: -1, startX: e.clientX, startY: e.clientY });
        return;
    }

    if (mode === 'waypoint') {
        if (control_points.length === 4) {
            const hitIdx = control_points.findIndex(p => Math.hypot(p.x - world.x, p.y - world.y) < 0.025 / transform.scale);
            if (hitIdx !== -1) {
                setDragInfo({ type: 'ctrl_point', index: hitIdx, startX: e.clientX, startY: e.clientY });
                return;
            }
        }
        const hitIdx = waypoints.findIndex(p => Math.hypot(p.x - world.x, p.y - world.y) < 0.025 / transform.scale);
        if (hitIdx !== -1) {
            setDragInfo({ type: 'point', index: hitIdx, startX: e.clientX, startY: e.clientY });
            return;
        }
        if (waypoints.length < 10 && e.button === 0) onUpdate({ waypoints: [...waypoints, world] });
    } else {
        if (selectedBBoxIdx !== undefined && selectedBBoxIdx !== null) {
            const b = bboxes[selectedBBoxIdx];
            const x = b.cx - b.w/2, y = b.cy - b.h/2, w = b.w, h = b.h;
            const hs = 0.02 / transform.scale;
            const check = (px: number, py: number) => Math.hypot(px - world.x, py - world.y) < hs;
            if (check(x, y)) { setDragInfo({ type: 'resize', index: selectedBBoxIdx, handle: 'tl', startX: e.clientX, startY: e.clientY }); return; }
            if (check(x+w, y)) { setDragInfo({ type: 'resize', index: selectedBBoxIdx, handle: 'tr', startX: e.clientX, startY: e.clientY }); return; }
            if (check(x, y+h)) { setDragInfo({ type: 'resize', index: selectedBBoxIdx, handle: 'bl', startX: e.clientX, startY: e.clientY }); return; }
            if (check(x+w, y+h)) { setDragInfo({ type: 'resize', index: selectedBBoxIdx, handle: 'br', startX: e.clientX, startY: e.clientY }); return; }
        }

        const hitBoxIdx = bboxes.reduce((bestIdx, b, idx) => {
            const isInside = Math.abs(b.cx - world.x) < b.w/2 && Math.abs(b.cy - world.y) < b.h/2;
            if (!isInside) return bestIdx;
            if (bestIdx === -1) return idx;
            return (bboxes[idx].w * bboxes[idx].h < bboxes[bestIdx].w * bboxes[bestIdx].h) ? idx : bestIdx;
        }, -1);

        if (hitBoxIdx !== -1) {
            onSelectBBox?.(hitBoxIdx);
            setDragInfo({ type: 'box', index: hitBoxIdx, startX: e.clientX, startY: e.clientY });
        } else {
            setIsDrawing(true);
            setStartPos(world);
            onSelectBBox?.(null);
        }
    }
  };

  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => {
        if (!dragInfo && !isDrawing) {
            // Still update mouseWorld for crosshairs
            const world = toWorld(e.clientX, e.clientY);
            setMouseWorld(world);
            return;
        }

        const world = toWorld(e.clientX, e.clientY);
        setMouseWorld(world);

        if (dragInfo?.type === 'pan') {
            handlePan(e.clientX - dragInfo.startX, e.clientY - dragInfo.startY);
            setDragInfo({ ...dragInfo, startX: e.clientX, startY: e.clientY });
        } else if (dragInfo?.type === 'ctrl_point') {
            const nextCtrls = [...control_points];
            nextCtrls[dragInfo.index] = world;
            const [p0, p1, p2, p3] = nextCtrls;
            const nextWaypoints: Waypoint[] = [];
            for (let i = 0; i < 10; i++) nextWaypoints.push(getBezierPoint(p0, p1, p2, p3, i / 9));
            onUpdate({ control_points: nextCtrls, waypoints: nextWaypoints });
        } else if (dragInfo?.type === 'point') {
            const newWps = [...waypoints];
            newWps[dragInfo.index] = world;
            onUpdate({ waypoints: newWps, control_points: [] });
        } else if (dragInfo?.type === 'resize') {
            const newBoxes = [...bboxes];
            const b = { ...newBoxes[dragInfo.index] };
            let x1 = b.cx - b.w/2, y1 = b.cy - b.h/2, x2 = b.cx + b.w/2, y2 = b.cy + b.h/2;
            if (dragInfo.handle === 'tl') { x1 = world.x; y1 = world.y; }
            if (dragInfo.handle === 'tr') { x2 = world.x; y1 = world.y; }
            if (dragInfo.handle === 'bl') { x1 = world.x; y2 = world.y; }
            if (dragInfo.handle === 'br') { x2 = world.x; y2 = world.y; }
            newBoxes[dragInfo.index] = { cx: (x1+x2)/2, cy: (y1+y2)/2, w: Math.abs(x2-x1), h: Math.abs(y2-y1), category: b.category };
            onUpdate({ bboxes: newBoxes });
        } else if (dragInfo?.type === 'box') {
            const newBoxes = [...bboxes];
            const box = newBoxes[dragInfo.index];
            const prevWorld = toWorld(dragInfo.startX, dragInfo.startY);
            const dx = world.x - prevWorld.x;
            const dy = world.y - prevWorld.y;
            newBoxes[dragInfo.index] = { ...box, cx: box.cx + dx, cy: box.cy + dy };
            onUpdate({ bboxes: newBoxes });
            setDragInfo({ ...dragInfo, startX: e.clientX, startY: e.clientY });
        }
    };

    const handleGlobalMouseUp = () => {
        if (isDrawing && startPos) {
            const w = Math.abs(mouseWorld.x - startPos.x);
            const h = Math.abs(mouseWorld.y - startPos.y);
            if (w > 0.001 && h > 0.001) {
                const newBox: BBox = { cx: (startPos.x + mouseWorld.x) / 2, cy: (startPos.y + mouseWorld.y) / 2, w, h, category: 0 };
                onUpdate({ bboxes: [...bboxes, newBox] });
                onSelectBBox?.(bboxes.length);
            }
        }
        setDragInfo(null);
        setIsDrawing(false);
        setStartPos(null);
    };

    window.addEventListener('mousemove', handleGlobalMouseMove);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => {
        window.removeEventListener('mousemove', handleGlobalMouseMove);
        window.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [dragInfo, isDrawing, startPos, mouseWorld, bboxes, waypoints, control_points, toWorld, handlePan, onUpdate, onSelectBBox]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const onWheel = (e: WheelEvent) => {
      // If the event target is inside a scrollable element, don't zoom
      const target = e.target as HTMLElement;
      if (target && target.closest('.overflow-y-auto')) {
          return; // Let the local scroll happen
      }

      e.preventDefault(); e.stopPropagation();
      const rect = container.getBoundingClientRect();
      handleZoom(e.deltaY, e.clientX - rect.left, e.clientY - rect.top);
    };
    // Removed capture: true to allow scrollable children to handle events first if needed,
    // though we use the 'target' check above for more precision.
    container.addEventListener('wheel', onWheel, { passive: false });
    return () => container.removeEventListener('wheel', onWheel);
  }, [handleZoom]);

  const filteredQuickClasses = useMemo(() => {
    return classNames.map((c, i) => ({ name: c, id: i }))
                     .filter(c => c.name.toLowerCase().includes(searchTerm.toLowerCase()))
                     .slice(0, 5);
  }, [classNames, searchTerm]);

  const selectedBoxPos = useMemo(() => {
    if (selectedBBoxIdx === undefined || selectedBBoxIdx === null || !bboxes[selectedBBoxIdx]) return null;
    const b = bboxes[selectedBBoxIdx];
    return fromWorld(b.cx + b.w/2, b.cy - b.h/2);
  }, [selectedBBoxIdx, bboxes, fromWorld]);

  return (
    <div ref={containerRef} className="relative w-full h-full bg-[#050505] overflow-hidden flex items-center justify-center rounded-2xl border border-white/20 shadow-inner select-none" style={{ overscrollBehavior: 'none', touchAction: 'none' }}>
      {img ? (
        <canvas ref={canvasRef} onMouseDown={onMouseDown} onContextMenu={e => e.preventDefault()} className="cursor-crosshair transition-opacity duration-300 block" />
      ) : (
        <div className="flex flex-col items-center gap-4">
            <div className="w-12 h-12 border-4 border-accent border-t-transparent rounded-full animate-spin" />
            <p className="font-cyber text-accent tracking-[0.2em] text-[10px] animate-pulse font-bold uppercase">Streaming_Neural_Data</p>
        </div>
      )}

      {/* Floating Quick Search Picker */}
      {selectedBoxPos && (
        <div
            className="absolute z-[60] flex flex-col gap-1 animate-in slide-in-from-top-2 duration-200"
            style={{
                left: selectedBoxPos.x + 40, // More offset to avoid blocking handles
                top: selectedBoxPos.y - 20
            }}
        >
            <div className="bg-[#0a0a0c] border-2 border-accent/50 rounded-xl shadow-[0_0_30px_rgba(0,0,0,0.8)] overflow-hidden min-w-[180px]">
                <div className="flex items-center px-3 py-2 bg-white/5 gap-2 border-b border-white/10">
                    <Search className="w-3.5 h-3.5 text-accent" />
                    <input
                        ref={quickSearchRef}
                        type="text"
                        placeholder="Search class..."
                        value={searchTerm}
                        onChange={(e) => setSearchSearchTerm(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && filteredQuickClasses.length > 0) {
                                const next = [...bboxes];
                                next[selectedBBoxIdx!] = { ...next[selectedBBoxIdx!], category: filteredQuickClasses[0].id };
                                onUpdate({ bboxes: next });
                                onSelectBBox?.(null);
                            }
                            if (e.key === 'Escape') onSelectBBox?.(null);
                        }}
                        className="bg-transparent border-none outline-none text-[11px] font-black text-white placeholder:text-white/20 w-full"
                    />
                </div>
                <div className="p-1 max-h-[150px] overflow-y-auto cyber-scrollbar">
                    {filteredQuickClasses.map((c) => (
                        <button
                            key={c.id}
                            onClick={() => {
                                const next = [...bboxes];
                                next[selectedBBoxIdx!] = { ...next[selectedBBoxIdx!], category: c.id };
                                onUpdate({ bboxes: next });
                                onSelectBBox?.(null);
                            }}
                            className="w-full flex items-center justify-between px-3 py-2 rounded-lg hover:bg-accent hover:text-black transition-all group"
                        >
                            <span className="text-[10px] font-black uppercase tracking-wider">{c.id}: {c.name}</span>
                            <Check className="w-3 h-3 opacity-0 group-hover:opacity-100" />
                        </button>
                    ))}
                </div>
            </div>
        </div>
      )}

      <div className="absolute top-6 left-6 flex gap-2">
         <div className="bg-black/90 backdrop-blur-xl border border-white/30 px-6 py-3 rounded-2xl flex items-center gap-4 shadow-2xl">
            <div className={`w-3.5 h-3.5 rounded-full ${mode === 'bbox' ? 'bg-cyan-400 shadow-[0_0_15px_#22d3ee]' : 'bg-green-400 shadow-[0_0_15px_#4ade80]'} animate-pulse`} />
            <span className="text-[14px] font-black text-white tracking-[0.2em] uppercase">{mode} MODE</span>
         </div>
      </div>
      <div className="absolute bottom-6 right-6 bg-black/90 backdrop-blur-xl border border-white/20 p-6 rounded-[2.5rem] space-y-3 shadow-2xl pointer-events-none min-w-[220px]">
          <div className="flex justify-between gap-8"><span className="text-[11px] font-black text-white/60 uppercase tracking-widest">Zoom</span><span className="text-[14px] text-accent font-black">{(transform.scale * 100).toFixed(0)}%</span></div>
          <div className="flex justify-between gap-8 border-t border-white/10 pt-3"><span className="text-[11px] font-black text-white/60 uppercase tracking-widest">T-Coordinate</span><span className="text-[13px] text-white font-mono font-black tracking-tighter">X:{mouseWorld.x.toFixed(3)} Y:{mouseWorld.y.toFixed(3)}</span></div>
      </div>
    </div>
  );
};
