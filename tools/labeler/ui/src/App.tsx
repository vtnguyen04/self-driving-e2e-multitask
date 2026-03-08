import React, { Suspense } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';

const ProjectsPage = React.lazy(() => import('./pages/ProjectsPage').then(m => ({ default: m.ProjectsPage })));
const DatasetPage = React.lazy(() => import('./pages/DatasetPage').then(m => ({ default: m.DatasetPage })));
const AnnotatePage = React.lazy(() => import('./pages/AnnotatePage').then(m => ({ default: m.AnnotatePage })));
const VersionsPage = React.lazy(() => import('./pages/VersionsPage').then(m => ({ default: m.VersionsPage })));
const AnalyticsPage = React.lazy(() => import('./pages/AnalyticsPage').then(m => ({ default: m.AnalyticsPage })));

const PageLoader = () => (
  <div className="h-screen w-screen flex items-center justify-center bg-black">
    <div className="w-12 h-12 border-4 border-white/10 border-t-[#00ff41] rounded-full animate-spin" />
  </div>
);

function App() {
  return (
    <BrowserRouter>
      <div className="h-screen w-screen overflow-hidden bg-black selection:bg-accent selection:text-black">
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<ProjectsPage />} />
            <Route path="/dataset/:id" element={<DatasetPage />} />
            <Route path="/annotate/:id" element={<AnnotatePage />} />
            <Route path="/versions/:id" element={<VersionsPage />} />
            <Route path="/analytics/:id" element={<AnalyticsPage />} />
          </Routes>
        </Suspense>
      </div>
    </BrowserRouter>
  );
}

export default App;
