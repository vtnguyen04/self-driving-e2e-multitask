import { Activity, ArrowLeft, BarChart3, Database, PieChart as PieChartIcon, Target } from 'lucide-react';
import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Bar, BarChart, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { API } from '../api';
import { MainLayout } from '../components/layout/MainLayout';

const COLORS = [
  '#00ff41', '#00f3ff', '#ffcc00', '#ff003c', '#ff00ff',
  '#00ffff', '#ffff00', '#ff8800', '#88ff00', '#00ff88',
  '#0088ff', '#8800ff', '#ff0088', '#ffffff'
];

interface AnalyticsData {
  class_distribution: { id: number, name: string, count: number }[];
  command_distribution: { id: number, name: string, count: number }[];
  total_samples: number;
  labeled_samples: number;
  total_bboxes: number;
  total_waypoints: number;
  samples_with_waypoints: number;
}

export const AnalyticsPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const projectId = id ? parseInt(id) : 1;
  const navigate = useNavigate();
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await API.labels.getAnalytics(projectId);
        setData(res);
      } catch (err) {
        console.error("Analytics load failed", err);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [projectId]);

  if (loading) return (
    <div className="h-screen bg-black flex items-center justify-center">
      <div className="flex flex-col items-center gap-6">
        <div className="w-16 h-16 border-4 border-accent border-t-transparent rounded-full animate-spin shadow-[0_0_20px_rgba(0,255,65,0.3)]" />
        <p className="font-cyber text-accent tracking-[0.5em] text-xs animate-pulse">ANALYZING_DATASET_VECTORS...</p>
      </div>
    </div>
  );

  if (!data) return null;

  const pieData = [
    { name: 'Labeled', value: data.labeled_samples },
    { name: 'Unlabeled', value: data.total_samples - data.labeled_samples },
  ];

  const progress = (data.labeled_samples / data.total_samples) * 100 || 0;

  return (
    <MainLayout>
      <div className="p-8 space-y-8 bg-[#050505] selection:bg-accent selection:text-black">

        {/* Header */}
        <div className="flex justify-between items-center bg-[#0a0a0c]/80 backdrop-blur-xl p-6 rounded-3xl border border-white/5 shadow-2xl sticky top-0 z-20">
          <div className="flex items-center gap-6">
            <button
              onClick={() => navigate(`/dataset/${projectId}`)}
              className="p-3 rounded-2xl bg-white/5 hover:bg-white/10 transition-all border border-white/10 group active:scale-90"
            >
              <ArrowLeft className="w-5 h-5 text-white/50 group-hover:text-white" />
            </button>
            <div>
              <h1 className="text-3xl font-black text-white flex items-center gap-4 font-cyber tracking-tight uppercase">
                <BarChart3 className="w-7 h-7 text-accent" />
                Dataset_Analytics
              </h1>
              <p className="text-white/40 text-[10px] font-mono uppercase tracking-[0.3em] mt-1 italic">NeuroPilot Intelligence Engine v4.0</p>
            </div>
          </div>
          <div className="flex items-center gap-8 pr-4">
             <div className="text-right">
                <p className="text-white/30 text-[9px] font-black uppercase tracking-widest mb-1">Health Score</p>
                <p className="text-accent font-cyber text-xl font-black">STABLE</p>
             </div>
             <div className="w-px h-10 bg-white/10" />
             <div className="text-right">
                <p className="text-white/30 text-[9px] font-black uppercase tracking-widest mb-1">Data Volume</p>
                <p className="text-white font-cyber text-xl font-black">{data.total_samples} <span className="text-[10px] text-white/40">SAMPLES</span></p>
             </div>
          </div>
        </div>

        {/* Summary Row */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="p-6 rounded-3xl bg-[#0a0a0c] border border-white/5 relative overflow-hidden group hover:border-accent/30 transition-all">
            <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
              <Database className="w-20 h-20 text-white" />
            </div>
            <p className="text-white/40 text-[10px] font-black uppercase tracking-widest mb-3">Total BBoxes</p>
            <p className="text-5xl font-black text-white font-cyber tracking-tighter">{data.total_bboxes}</p>
            <div className="mt-4 flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
              <p className="text-accent/60 text-[10px] font-bold uppercase tracking-wider">Object detection active</p>
            </div>
          </div>
          <div className="p-6 rounded-3xl bg-[#0a0a0c] border border-white/5 relative overflow-hidden group hover:border-accent/30 transition-all">
            <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
              <Target className="w-20 h-20 text-white" />
            </div>
            <p className="text-white/40 text-[10px] font-black uppercase tracking-widest mb-3">Avg BBoxes / Image</p>
            <p className="text-5xl font-black text-white font-cyber tracking-tighter">
              {(data.total_bboxes / (data.labeled_samples || 1)).toFixed(1)}
            </p>
            <p className="mt-4 text-white/30 text-[10px] font-bold uppercase tracking-wider">Across labeled samples</p>
          </div>
          <div className="p-6 rounded-3xl bg-[#0a0a0c] border border-white/5 relative overflow-hidden group hover:border-accent/30 transition-all">
            <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
              <Activity className="w-20 h-20 text-white" />
            </div>
            <p className="text-white/40 text-[10px] font-black uppercase tracking-widest mb-3">Total Waypoints</p>
            <p className="text-5xl font-black text-white font-cyber tracking-tighter">{data.total_waypoints}</p>
            <p className="mt-4 text-white/30 text-[10px] font-bold uppercase tracking-wider">In {data.samples_with_waypoints} samples</p>
          </div>
          <div className="p-6 rounded-3xl bg-[#0a0a0c] border border-white/5 relative overflow-hidden group hover:border-accent/30 transition-all">
            <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
              <PieChartIcon className="w-20 h-20 text-white" />
            </div>
            <p className="text-white/40 text-[10px] font-black uppercase tracking-widest mb-3">Labeling Progress</p>
            <p className="text-5xl font-black text-accent font-cyber tracking-tighter">{progress.toFixed(0)}%</p>
            <div className="mt-4 w-full h-1 bg-white/10 rounded-full overflow-hidden">
                <div className="h-full bg-accent" style={{ width: `${progress}%` }} />
            </div>
          </div>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-[500px]">

          {/* Class Distribution Bar Chart */}
          <div className="p-8 rounded-[40px] bg-[#0a0a0c] border border-white/5 flex flex-col shadow-2xl">
            <div className="flex justify-between items-center mb-10">
              <h3 className="text-white text-sm font-black uppercase tracking-[0.2em] font-cyber flex items-center gap-3">
                <BarChart3 className="w-4 h-4 text-accent" />
                Class_Balance_Distribution
              </h3>
              <div className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-[9px] text-white/40 font-black uppercase tracking-widest">
                Sorted by Count
              </div>
            </div>
            <div className="w-full h-[400px]">
              <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                <BarChart
                  data={data.class_distribution}
                  layout="vertical"
                  margin={{ left: 20, right: 40, top: 0, bottom: 0 }}
                >
                  <XAxis type="number" hide />
                  <YAxis
                    dataKey="name"
                    type="category"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#fff', fontSize: 10, fontWeight: 900, fontFamily: 'monospace' }}
                    width={100}
                  />
                  <Tooltip
                    cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                    contentStyle={{ backgroundColor: '#0a0a0c', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '16px', fontSize: '10px' }}
                  />
                  <Bar dataKey="count" radius={[0, 8, 8, 0]} barSize={24}>
                    {data.class_distribution.map((entry, index) => (
                      <Cell key={index} fill={COLORS[entry.id % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Dataset Completion Pie Chart */}
          <div className="p-8 rounded-[40px] bg-[#0a0a0c] border border-white/5 flex flex-col shadow-2xl relative overflow-hidden">
             <div className="absolute inset-0 bg-gradient-to-br from-accent/5 to-transparent pointer-events-none" />
             <div className="flex justify-between items-center mb-10 relative z-10">
              <h3 className="text-white text-sm font-black uppercase tracking-[0.2em] font-cyber flex items-center gap-3">
                <PieChartIcon className="w-4 h-4 text-accent" />
                Annotation_Coverage
              </h3>
            </div>
            <div className="w-full h-[400px] relative z-10">
              <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={80}
                    outerRadius={140}
                    paddingAngle={8}
                    dataKey="value"
                    stroke="none"
                  >
                    <Cell fill="#00ff41" />
                    <Cell fill="rgba(255,255,255,0.05)" />
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0a0a0c', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '16px', fontSize: '10px' }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none">
                  <p className="text-xs text-white/30 font-black uppercase tracking-widest mb-1">Global</p>
                  <p className="text-5xl font-black text-white font-cyber tracking-tighter">{progress.toFixed(0)}%</p>
                  <p className="text-[10px] text-accent font-black uppercase tracking-[0.2em] mt-2">Verified</p>
              </div>
            </div>
          </div>

        </div>

        {/* Behavioral Logic Charts Row */}
        <div className="grid grid-cols-1 gap-8 h-[400px]">

          {/* Command Distribution Bar Chart */}
          <div className="p-8 rounded-[40px] bg-[#0a0a0c] border border-white/5 flex flex-col shadow-2xl overflow-hidden group hover:border-accent/20 transition-all">
            <div className="flex justify-between items-center mb-8">
              <h3 className="text-white text-sm font-black uppercase tracking-[0.2em] font-cyber flex items-center gap-3">
                <Activity className="w-4 h-4 text-accent" />
                Behavioral_Command_Diversity
              </h3>
              <div className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-[9px] text-white/40 font-black uppercase tracking-widest">
                Drive Logic Analysis
              </div>
            </div>
            <div className="w-full h-[300px]">
              <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                <BarChart
                  data={data.command_distribution}
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                >
                  <XAxis
                    dataKey="name"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#888', fontSize: 10, fontWeight: 900, fontFamily: 'monospace' }}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#888', fontSize: 10, fontWeight: 900, fontFamily: 'monospace' }}
                  />
                  <Tooltip
                    cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                    contentStyle={{ backgroundColor: '#0a0a0c', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '16px', fontSize: '10px' }}
                  />
                  <Bar dataKey="count" radius={[8, 8, 0, 0]} barSize={60}>
                    {data.command_distribution.map((_entry, index) => (
                      <Cell key={index} fill={index === 0 ? '#00ff41' : '#00f3ff'} fillOpacity={0.8} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>

        {/* Detail List */}
        <div className="bg-[#0a0a0c] rounded-[40px] border border-white/5 p-8 shadow-2xl overflow-hidden mb-12">
            <div className="flex items-center gap-4 mb-8">
                <div className="w-1.5 h-6 bg-accent rounded-full" />
                <h3 className="text-white text-sm font-black uppercase tracking-[0.2em] font-cyber">Raw_Metrics_Log</h3>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-12">
                {[
                    { label: 'Samples Analyzed', value: data.total_samples },
                    { label: 'Successful Labels', value: data.labeled_samples },
                    { label: 'Bounding Box Density (Avg)', value: (data.total_bboxes / (data.labeled_samples || 1)).toFixed(2) },
                    { label: 'Waypoint Complexity', value: (data.total_waypoints / (data.samples_with_waypoints || 1)).toFixed(2) },
                    { label: 'Classes Defined', value: data.class_distribution.length },
                    { label: 'Missing Annotations', value: data.total_samples - data.labeled_samples },
                    { label: 'Trajectory Coverage', value: `${((data.samples_with_waypoints / data.total_samples) * 100).toFixed(1)}%` },
                    { label: 'Sync Status', value: 'LIVE' },
                ].map((item, i) => (
                    <div key={i} className="space-y-2 group">
                        <p className="text-white/30 text-[9px] font-black uppercase tracking-widest font-mono group-hover:text-white/50 transition-colors">{item.label}</p>
                        <p className="text-xl font-black text-white font-cyber tracking-tight group-hover:text-accent transition-colors">{item.value}</p>
                    </div>
                ))}
            </div>
        </div>

      </div>
    </MainLayout>
  );
};
