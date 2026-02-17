import { Calendar, Database, Download, History as HistoryIcon, Plus, Trash2 } from 'lucide-react';
import React, { useCallback, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { API } from '../api';
import { MainLayout } from '../components/layout/MainLayout';

export const VersionsPage: React.FC = () => {
  const [versions, setVersions] = useState<any[]>([]);
  const [isPublishing, setIsPublishing] = useState(false);

  const { id } = useParams<{ id: string }>();
  const projectId = id ? parseInt(id) : 1;

  const loadVersions = useCallback(() => {
    API.labels.getVersions(projectId).then(setVersions);
  }, [projectId]);

  useEffect(() => {
    loadVersions();
  }, [loadVersions]);

  const handlePublish = async () => {
    const name = prompt("Enter version name (e.g. v1, final_run):");
    if (!name) return;

    setIsPublishing(true);
    try {
        await API.labels.publish(projectId, name);
        loadVersions();
        alert("Physical Export Complete! Check the 'exports' directory.");
    } catch (e) {
        alert("Publishing failed.");
    } finally {
        setIsPublishing(false);
    }
  };

  return (
    <MainLayout>
      <div className="p-10 max-w-6xl mx-auto space-y-10">
        <header className="flex justify-between items-center">
            <div>
                <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Versions History</h1>
                <p className="text-white/40">Snapshots of your dataset ready for training.</p>
            </div>
            <button
                onClick={handlePublish}
                disabled={isPublishing}
                className="bg-accent text-black px-6 py-2.5 rounded-lg font-bold flex items-center gap-2 hover:bg-white transition-all disabled:opacity-50"
            >
                <Plus className="w-5 h-5" /> {isPublishing ? "Publishing..." : "Generate New Version"}
            </button>
        </header>

        <div className="space-y-4">
            {versions.length > 0 ? versions.map((v, i) => (
                <div key={i} className="bg-white/5 border border-white/10 rounded-2xl p-6 flex items-center gap-8 group hover:bg-white/10 transition-all">
                    <div className="w-16 h-16 bg-accent/20 rounded-2xl flex flex-col items-center justify-center text-accent">
                        <span className="text-xs font-cyber">v{v.id || i+1}</span>
                        <Database className="w-5 h-5" />
                    </div>

                    <div className="flex-1">
                        <div className="flex items-center gap-3 mb-1">
                            <h3 className="text-lg font-bold text-white">{v.name}</h3>
                            <span className="text-[10px] px-2 py-0.5 bg-white/5 rounded border border-white/10 text-white/40 font-cyber">STRETCH TO 640x640</span>
                        </div>
                        <div className="flex gap-6 text-xs text-white/40">
                             <span className="flex items-center gap-2"><Calendar className="w-3.5 h-3.5" /> Jan 28, 2026</span>
                             <span className="flex items-center gap-2"><Database className="w-3.5 h-3.5" /> {v.sample_count || 3912} Images</span>
                        </div>
                    </div>

                    <div className="flex gap-2">
                        <button
                            onClick={() => window.open(`/api/v1/labels/projects/${projectId}/versions/${v.id}/download`)}
                            className="p-3 bg-white/5 rounded-xl hover:bg-accent hover:text-black transition-all group-hover:scale-105"
                        >
                            <Download className="w-5 h-5" />
                        </button>
                        <button className="p-3 bg-white/5 rounded-xl hover:bg-red-500/20 hover:text-red-500 transition-all">
                            <Trash2 className="w-5 h-5" />
                        </button>
                    </div>
                </div>
            )) : (
                <div className="h-64 border-2 border-dashed border-white/5 rounded-3xl flex flex-col items-center justify-center text-white/10">
                    <HistoryIcon className="w-16 h-16 mb-4" />
                    <p className="font-cyber tracking-widest text-sm uppercase">No history pulses detected</p>
                </div>
            )}
        </div>
      </div>
    </MainLayout>
  );
};
