import { Info, Play, X } from 'lucide-react';
import React, { useState } from 'react';

interface PublishModalProps {
  isOpen: boolean;
  onClose: () => void;
  onPublish: (name: string, train: number, val: number, test: number) => void;
  isPublishing: boolean;
}

export const PublishModal: React.FC<PublishModalProps> = ({ isOpen, onClose, onPublish, isPublishing }) => {
  const [name, setName] = useState('v1');
  const [ratios, setRatios] = useState({ train: 0.8, val: 0.1, test: 0.1 });

  if (!isOpen) return null;

  const handleRatioChange = (key: 'train' | 'val' | 'test', value: number) => {
    const newValue = Math.max(0, Math.min(1, value));
    const remaining = 1 - newValue;
    const otherKeys = (['train', 'val', 'test'] as const).filter(k => k !== key);

    // Proportional distribution
    const otherTotal = ratios[otherKeys[0]] + ratios[otherKeys[1]];
    let d0 = 0;
    let d1 = 0;

    if (otherTotal > 0) {
        d0 = (ratios[otherKeys[0]] / otherTotal) * remaining;
        d1 = (ratios[otherKeys[1]] / otherTotal) * remaining;
    } else {
        d0 = remaining / 2;
        d1 = remaining / 2;
    }

    setRatios({
        ...ratios,
        [key]: newValue,
        [otherKeys[0]]: d0,
        [otherKeys[1]]: d1
    });
  };

  const total = ratios.train + ratios.val + ratios.test;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/80 backdrop-blur-sm animate-in fade-in duration-300">
      <div className="w-full max-w-md bg-[#0a0a0c] border border-white/10 rounded-3xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-300">
        <header className="px-8 py-6 border-b border-white/5 flex justify-between items-center bg-white/[0.02]">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">Generate Version</h2>
            <p className="text-white/40 text-xs">Create a training-ready snapshot.</p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full transition-colors text-white/40 hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </header>

        <div className="p-8 space-y-8">
          {/* Name Input */}
          <div className="space-y-3">
            <label className="text-[10px] font-cyber uppercase tracking-widest text-accent flex items-center gap-2">
              Version Identifier <Info className="w-3 h-3 text-white/20" />
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. baseline_run, v2"
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-white/10 focus:outline-none focus:border-accent/40 focus:ring-1 focus:ring-accent/40 transition-all font-mono"
            />
          </div>

          {/* Ratio Sliders */}
          <div className="space-y-6">
            <label className="text-[10px] font-cyber uppercase tracking-widest text-accent">Dataset Split Ratios</label>

            <div className="space-y-6">
              {[
                { label: 'Train', key: 'train', color: 'bg-accent' },
                { label: 'Validation', key: 'val', color: 'bg-blue-500' },
                { label: 'Test', key: 'test', color: 'bg-purple-500' }
              ].map((item) => (
                <div key={item.key} className="space-y-2">
                  <div className="flex justify-between text-xs font-medium">
                    <span className="text-white/60">{item.label}</span>
                    <span className="text-accent font-cyber">{(ratios[item.key as keyof typeof ratios] * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={ratios[item.key as keyof typeof ratios]}
                    onChange={(e) => handleRatioChange(item.key as any, parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-white/5 rounded-full appearance-none cursor-pointer accent-accent"
                  />
                </div>
              ))}
            </div>

            {/* Visual Progress Bar */}
            <div className="h-4 w-full bg-white/5 rounded-full overflow-hidden flex border border-white/5 p-0.5">
                <div style={{ width: `${ratios.train * 100}%` }} className="h-full bg-accent rounded-l-full transition-all duration-500"></div>
                <div style={{ width: `${ratios.val * 100}%` }} className="h-full bg-blue-500 transition-all duration-500"></div>
                <div style={{ width: `${ratios.test * 100}%` }} className="h-full bg-purple-500 rounded-r-full transition-all duration-500"></div>
            </div>
            {Math.abs(total - 1) > 0.01 && (
                <p className="text-[10px] text-red-500/60 font-medium">Warning: Ratios must sum to 100% (Current: {(total * 100).toFixed(0)}%)</p>
            )}
          </div>
        </div>

        <footer className="px-8 py-6 bg-white/[0.02] border-t border-white/5 flex gap-4">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-3 rounded-xl border border-white/10 text-white font-bold hover:bg-white/5 transition-all"
          >
            Cancel
          </button>
          <button
            onClick={() => onPublish(name, ratios.train, ratios.val, ratios.test)}
            disabled={isPublishing || Math.abs(total - 1) > 0.01}
            className="flex-[2] px-4 py-3 rounded-xl bg-accent text-black font-bold hover:bg-white transition-all disabled:opacity-50 flex items-center justify-center gap-2"
          >
            <Play className="w-4 h-4 fill-current" /> {isPublishing ? "Processing..." : "Start Export"}
          </button>
        </footer>
      </div>
    </div>
  );
};
