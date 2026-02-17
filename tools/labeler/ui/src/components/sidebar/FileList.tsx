import { clsx, type ClassValue } from 'clsx';
import { CheckCircle2, Circle, Copy, RotateCcw, Trash2 } from 'lucide-react';
import React from 'react';
import { twMerge } from 'tailwind-merge';
import { API } from '../../api';
import { Sample } from '../../types';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface FileListProps {
  samples: Sample[];
  selectedFilename: string | null;
  onSelect: (filename: string) => void;
  onUpdate: () => void;
}

export const FileList: React.FC<FileListProps> = ({ samples, selectedFilename, onSelect, onUpdate }) => {
  return (
    <div className="flex flex-col gap-1">
      <h3 className="text-xs font-bold text-text-secondary uppercase tracking-widest mb-3">
        Dataset Samples ({samples.length})
      </h3>
      <div className="space-y-1 overflow-y-auto max-h-[calc(100vh-250px)]">
        {samples.map((sample) => (
          <div
            key={sample.filename}
            className={cn(
              "group relative flex items-center justify-between p-2.5 rounded-lg cursor-pointer transition-all border border-transparent",
              selectedFilename === sample.filename
                ? "bg-white/5 border-accent/20 text-white shadow-lg"
                : "text-white/40 hover:bg-white/5 hover:text-white"
            )}
          >
            <div onClick={() => onSelect(sample.filename)} className="flex-1 flex items-center gap-2 truncate">
                {sample.is_labeled ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-accent" />
                ) : (
                <Circle className="w-3.5 h-3.5 text-white/10" />
                )}
                <span className="text-[11px] truncate">{sample.filename}</span>
            </div>

            <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                 <button onClick={(e) => {
                    e.stopPropagation();
                    if (confirm("Reset labels for this image?")) API.labels.reset(sample.filename).then(onUpdate);
                 }} className="p-1 hover:text-accent transition-colors">
                    <RotateCcw className="w-3 h-3" />
                 </button>
                 <button onClick={(e) => {
                    e.stopPropagation();
                    API.labels.duplicate(sample.filename, `copy_${Date.now()}_${sample.filename}`).then(onUpdate);
                 }} className="p-1 hover:text-accent transition-colors">
                    <Copy className="w-3 h-3" />
                 </button>
                 <button onClick={(e) => {
                    e.stopPropagation();
                    if (confirm("Delete this sample?")) API.labels.delete(sample.filename).then(onUpdate);
                 }} className="p-1 hover:text-neon-red transition-colors">
                    <Trash2 className="w-3 h-3" />
                 </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
