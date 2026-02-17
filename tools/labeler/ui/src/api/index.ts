import { BBox, Sample, Version, Waypoint } from '../types';

const BASE_URL = '/api/v1';

async function fetchJSON<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...options.headers,
    },
  });
  if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
  return response.json();
}

export const API = {
  labels: {
    list: (params: { limit?: number; offset?: number; is_labeled?: boolean; split?: string; project_id?: number; class_id?: number }) => {
      const query = new URLSearchParams();
      if (params.limit) query.append('limit', params.limit.toString());
      if (params.offset) query.append('offset', params.offset.toString());
      if (params.is_labeled !== undefined) query.append('is_labeled', params.is_labeled.toString());
      if (params.split) query.append('split', params.split);
      if (params.project_id) query.append('project_id', params.project_id.toString());
      if (params.class_id !== undefined) query.append('class_id', params.class_id.toString());
      return fetchJSON<Sample[]>(`/labels/?${query.toString()}`);
    },
    getStats: (projectId?: number) => {
      const query = projectId ? `?project_id=${projectId}` : '';
      return fetchJSON<{ raw: number; train: number; val: number; test: number; labeled: number; total: number }>(`/labels/stats${query}`);
    },
    uploadData: (projectId: number, files: FileList) => {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }
      return fetchJSON(`/labels/projects/${projectId}/upload`, { method: 'POST', body: formData, headers: {} }); // Let browser set Content-Type for boundary
    },
    getProjects: () => fetchJSON<any[]>('/labels/projects'),
    deleteProject: (projectId: number) => fetchJSON(`/labels/projects/${projectId}`, { method: 'DELETE' }),
    getClasses: (projectId: number) => fetchJSON<string[]>(`/labels/projects/${projectId}/classes`),
    updateClasses: (projectId: number, classes: string[]) => fetchJSON(`/labels/projects/${projectId}/classes`, { method: 'POST', body: JSON.stringify(classes) }),
    getVersions: (projectId: number) => fetchJSON<any[]>(`/labels/projects/${projectId}/versions`),
    publish: (projectId: number, versionName: string) => fetchJSON(`/labels/projects/${projectId}/publish?version_name=${versionName}`, { method: 'POST' }),
    get: (filename: string) => fetchJSON<Sample>(`/labels/${filename}`),
    save: (filename: string, data: { bboxes: BBox[]; waypoints: Waypoint[]; command: number }) =>
      fetchJSON(`/labels/${filename}`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    delete: (filename: string) =>
      fetchJSON(`/labels/${filename}`, {
        method: 'DELETE',
      }),
    reset: (filename: string) =>
      fetchJSON(`/labels/${filename}/reset`, {
        method: 'POST',
      }),
    duplicate: (filename: string, newFilename: string) =>
      fetchJSON(`/labels/${filename}/duplicate?new_filename=${newFilename}`, {
        method: 'POST',
      }),
  },
  versions: {
    list: () => fetchJSON<Version[]>('/versions/'),
    publish: (name: string, description: string) =>
      fetchJSON<Version>('/versions/', {
        method: 'POST',
        body: JSON.stringify({ name, description }),
      }),
  },
};
