export interface BBox {
  cx: number;
  cy: number;
  w: number;
  h: number;
  category: number;
  id?: string; // For selection/edit
}

export interface Waypoint {
  x: number;
  y: number;
}

export interface Sample {
  filename: string;
  is_labeled: boolean;
  updated_at: string;
  command?: number;
  bboxes?: BBox[];
  waypoints?: Waypoint[];
  control_points?: Waypoint[];
  data?: string;
}

export interface Version {
  id: number;
  project_id: number;
  name: string;
  description?: string;
  created_at: string;
  sample_count: number;
  path: string;
}

export interface Project {
  id: number;
  name: string;
  description?: string;
  classes: string[];
  created_at: string;
}

export interface Stats {
  raw: number;
  train: number;
  val: number;
  test: number;
  labeled: number;
  total: number;
}
