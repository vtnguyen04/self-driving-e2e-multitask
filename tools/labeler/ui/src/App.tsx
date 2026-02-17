import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { AnnotatePage } from './pages/AnnotatePage';
import { DatasetPage } from './pages/DatasetPage';
import { ProjectsPage } from './pages/ProjectsPage';
import { VersionsPage } from './pages/VersionsPage';

function App() {
  return (
    <BrowserRouter>
      <div className="h-screen w-screen overflow-hidden bg-black selection:bg-accent selection:text-black">
        <Routes>
          <Route path="/" element={<ProjectsPage />} />
          <Route path="/dataset/:id" element={<DatasetPage />} />
          <Route path="/annotate/:id" element={<AnnotatePage />} />
          <Route path="/versions/:id" element={<VersionsPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
