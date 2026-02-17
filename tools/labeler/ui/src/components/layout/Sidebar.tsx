import { clsx, type ClassValue } from 'clsx';
import { BarChart2, BoxSelect, HelpCircle, History, Home, Layout, Settings } from 'lucide-react';
import React from 'react';
import { NavLink, useParams } from 'react-router-dom';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const Sidebar: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const projectId = id;

  const globalItems = [
    { icon: Home, label: 'Projects', path: '/' },
  ];

  const projectItems = [
    { icon: BarChart2, label: 'Dataset', path: `/dataset/${projectId}` },
    { icon: BoxSelect, label: 'Annotate', path: `/annotate/${projectId}` },
    { icon: History, label: 'Versions', path: `/versions/${projectId}` },
  ];

  return (
    <div className="w-64 bg-[#0a0a0c] border-r border-white/5 flex flex-col h-full font-rajdhani">
      <div className="p-6 flex items-center gap-3">
        <div className="w-8 h-8 bg-accent rounded flex items-center justify-center">
            <Layout className="w-5 h-5 text-black" />
        </div>
        <span className="text-xl font-bold tracking-tighter text-white">Labeler Pro</span>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {/* Global Hub Header - if in project */}
        {projectId && (
           <NavLink
             to="/"
             className="flex items-center gap-3 px-4 py-2 mb-6 text-white/20 hover:text-white transition-all group border border-white/5 rounded-xl hover:border-white/20"
           >
             <Home className="w-4 h-4" />
             <span className="text-[10px] font-bold tracking-widest uppercase">Global Projects</span>
           </NavLink>
        )}

        {/* Dynamic Items */}
        {(projectId ? projectItems : globalItems).map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => cn(
              "flex items-center gap-3 px-4 py-3 rounded-lg transition-all group",
              isActive
                ? "bg-accent/10 text-accent font-bold"
                : "text-white/40 hover:text-white/80 hover:bg-white/5"
            )}
          >
            <item.icon className={cn(
              "w-5 h-5 transition-transform group-hover:scale-110",
              "inherit"
            )} />
            <span className="text-sm font-medium">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-white/5 space-y-1">
        <NavItem icon={HelpCircle} label="Help & Docs" />
        <NavItem icon={Settings} label="Settings" />

        <div className="mt-4 p-4 rounded-xl bg-gradient-to-br from-accent/20 to-transparent border border-accent/20">
            <p className="text-[10px] text-accent font-cyber mb-1">FREE PLAN</p>
            <div className="h-1 bg-white/10 rounded-full overflow-hidden mb-2">
                <div className="h-full bg-accent w-2/3" />
            </div>
            <p className="text-[9px] text-white/40">Workspace capacity: 67%</p>
            <button className="w-full mt-3 py-2 bg-accent text-black text-[10px] font-bold rounded-lg hover:bg-white transition-colors">
                UPGRADE
            </button>
        </div>
      </div>
    </div>
  );
};

const NavItem = ({ icon: Icon, label }: { icon: any, label: string }) => (
    <button className="w-full flex items-center gap-3 px-4 py-2 text-white/40 hover:text-white/80 transition-colors">
        <Icon className="w-4 h-4" />
        <span className="text-sm font-medium">{label}</span>
    </button>
)
