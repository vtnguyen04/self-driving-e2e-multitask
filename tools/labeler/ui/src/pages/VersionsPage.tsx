import { Calendar, Database, Download, History as HistoryIcon, Plus, Trash2 } from 'lucide-react';
import React, { useCallback, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { API } from '../api';
import { MainLayout } from '../components/layout/MainLayout';
import { PublishModal } from '../components/PublishModal';
import { Version } from '../types';

export const VersionsPage: React.FC = () => {
  const [versions, setVersions] = useState<Version[]>([]);
  const [isPublishing, setIsPublishing] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const { id } = useParams<{ id: string }>();
  const projectId = id ? parseInt(id) : 1;

  const loadVersions = useCallback(() => {
    API.labels.getVersions(projectId).then(setVersions);
  }, [projectId]);

  useEffect(() => {
    loadVersions();
  }, [loadVersions]);

  const handlePublish = async (name: string, train: number, val: number, test: number) => {
    setIsPublishing(true);
    try {
        await API.labels.publish(projectId, name, train, val, test);
        loadVersions();
        setIsModalOpen(false);
        alert("Physical Export Complete! Check the 'data/versions' directory.");
    } catch (error) {
        console.error("Publishing failed:", error);
        alert("Publishing failed.");
    } finally {
        setIsPublishing(false);
    }
  };

  const handleDelete = async (versionId: number) => {
    if (!confirm("Are you sure you want to delete this version? All exported files will be removed.")) return;

    try {
        await API.labels.deleteVersion(versionId);
        loadVersions();
    } catch (error) {
        console.error("Deletion failed:", error);
        alert("Deletion failed.");
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
                onClick={() => setIsModalOpen(true)}
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
                            <span className="text-[10px] px-2 py-0.5 bg-white/5 rounded border border-white/10 text-white/40 font-cyber">YOLO FORMAT</span>
                        </div>
                        <div className="flex gap-6 text-xs text-white/40">
                             <span className="flex items-center gap-2"><Calendar className="w-3.5 h-3.5" /> {new Date(v.created_at).toLocaleDateString()}</span>
                             <span className="flex items-center gap-2"><Database className="w-3.5 h-3.5" /> {v.sample_count} Images</span>
                        </div>
                    </div>

                    <div className="flex gap-2">
                        <a
                            href={API.labels.downloadVersion(v.id)}
                            className="p-3 bg-white/5 rounded-xl hover:bg-accent hover:text-black transition-all group-hover:scale-105"
                        >
                            <Download className="w-5 h-5" />
                        </a>
                        <button
                            onClick={() => handleDelete(v.id)}
                            className="p-3 bg-white/5 rounded-xl hover:bg-red-500/20 hover:text-red-500 transition-all"
                        >
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

      <PublishModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onPublish={handlePublish}
        isPublishing={isPublishing}
      />
    </MainLayout>
  );
};
