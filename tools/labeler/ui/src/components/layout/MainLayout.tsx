import React from 'react';
import { Sidebar } from './Sidebar';

interface MainLayoutProps {
  children: React.ReactNode;
  inspector?: React.ReactNode;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ children, inspector }) => {
  return (
    <div className="flex h-screen bg-[#050505] text-[#e0e0e0] font-rajdhani overflow-hidden selection:bg-accent selection:text-black">
      {/* Global Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <main className="flex-1 relative flex flex-col overflow-hidden bg-[#050505]">
          <div className="flex-1 overflow-auto bg-[#0a0a0c]">
            {children}
          </div>
      </main>

      {/* Optional Inspector Panel */}
      {inspector && (
        <aside className="w-[320px] border-l border-white/5 bg-[#0a0a0c] p-6 overflow-y-auto cyber-scrollbar">
          {inspector}
        </aside>
      )}
    </div>
  );
};
