import { Folder, Plus, Search, Trash2 } from 'lucide-react';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { API } from '../api';
import { MainLayout } from '../components/layout/MainLayout';

export const ProjectsPage: React.FC = () => {
  const [projects, setProjects] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const navigate = useNavigate();

  useEffect(() => {
    API.labels.getProjects().then(setProjects);
    API.labels.getStats().then(setStats);
  }, []);

  const handleCreateProject = async () => {
    const name = prompt("Enter project name:");
    if (!name) return;
    try {
        await fetch('/api/v1/labels/projects', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, description: "New Dataset Project" })
        });
        API.labels.getProjects().then(setProjects);
    } catch (e) {
        alert("Creation failed.");
    }
  };

  return (
    <MainLayout>
      <div className="p-10 max-w-6xl mx-auto space-y-10">
        <header className="flex justify-between items-center">
            <div>
                <h1 className="text-3xl font-bold tracking-tight text-white mb-2">My Projects</h1>
                <p className="text-white/40">Manage your computer vision projects and datasets.</p>
            </div>
            <button
                onClick={handleCreateProject}
                className="bg-accent text-black px-6 py-2.5 rounded-lg font-bold flex items-center gap-2 hover:bg-white transition-all shadow-[0_0_20px_rgba(0,255,65,0.2)]"
            >
                <Plus className="w-5 h-5" /> New Project
            </button>
        </header>

        <div className="flex gap-4 items-center bg-white/5 p-2 rounded-xl border border-white/5">
            <div className="flex-1 flex items-center gap-3 px-3">
                <Search className="w-5 h-5 text-white/20" />
                <input placeholder="Search projects..." className="bg-transparent border-none focus:ring-0 text-sm w-full outline-none" />
            </div>
            <div className="h-4 w-[1px] bg-white/10" />
            <button className="px-4 py-2 text-xs font-medium text-white/40">Sort: Date Edited</button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((proj) => (
                <ProjectCard
                    key={proj.id}
                    title={proj.name}
                    type="Object Detection"
                    images={stats?.total || 0}
                    models={0}
                    updated="Just now"
                    onClick={() => navigate(`/dataset/${proj.id}`)}
                    onDelete={(e: any) => {
                        e.stopPropagation();
                        if (confirm(`Delete project "${proj.name}"? This will remove all associated labels.`)) {
                            API.labels.deleteProject(proj.id).then(() => {
                                API.labels.getProjects().then(setProjects);
                            });
                        }
                    }}
                />
            ))}
             <div className="border-2 border-dashed border-white/5 rounded-2xl flex flex-col items-center justify-center p-8 text-white/10 hover:text-white/20 hover:border-white/10 transition-all cursor-pointer group">
                <Plus className="w-12 h-12 mb-4 group-hover:scale-110 transition-transform" />
                <span className="font-bold tracking-widest text-xs">CREATE NEW</span>
             </div>
        </div>
      </div>
    </MainLayout>
  );
};

const ProjectCard = ({ title, type, images, models, updated, active, onClick, onDelete }: any) => (
    <div
        onClick={onClick}
        className={cn(
            "bg-white/5 border border-white/10 rounded-2xl p-6 transition-all cursor-pointer group hover:bg-white/[0.08] hover:border-white/20 hover:translate-y-[-4px] relative",
            active && "border-accent/40 bg-accent/[0.02]"
        )}
    >
        <button
            onClick={(e) => { e.stopPropagation(); onDelete(e); }}
            className="absolute top-4 right-4 p-2 text-white/10 hover:text-neon-red hover:bg-neon-red/10 rounded-lg transition-all opacity-0 group-hover:opacity-100 z-20"
        >
            <Trash2 className="w-4 h-4" />
        </button>
        <div className="flex justify-between items-start mb-6">
            <div className="w-12 h-12 bg-accent/20 rounded-xl flex items-center justify-center">
                <Folder className="w-6 h-6 text-accent" />
            </div>
            <span className="text-[10px] font-cyber text-accent/60 px-2 py-0.5 rounded border border-accent/20 uppercase tracking-tighter">
                {type}
            </span>
        </div>
        <h3 className="text-lg font-bold text-white mb-4 transition-colors group-hover:text-accent">{title}</h3>
        <div className="flex items-center gap-4 text-xs text-white/40 mb-6">
            <span>{images.toLocaleString()} Images</span>
            <div className="w-1 h-1 rounded-full bg-white/10" />
            <span>{models} Models</span>
        </div>
        <div className="pt-4 border-t border-white/5 flex justify-between items-center">
            <span className="text-[10px] text-white/20 uppercase tracking-widest">Edited {updated}</span>
            <div className="flex -space-x-2">
                {[1,2,3].map(i => (
                    <div key={i} className="w-6 h-6 rounded-full border-2 border-[#0a0a0c] bg-white/10 flex items-center justify-center text-[8px] text-white/40">U{i}</div>
                ))}
            </div>
        </div>
    </div>
)

const cn = (...classes: any) => classes.filter(Boolean).join(' ');
