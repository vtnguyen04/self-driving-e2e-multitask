import cv2
import numpy as np
from pathlib import Path

def generate_lane_mask(img_shape, waypoints, lane_width=80):
    """
    Generate a filled polygon representing the drivable lane based on waypoints.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    left_boundary = []
    right_boundary = []

    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i+1]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            nx = -dy / length
            ny = dx / length
        else:
            nx, ny = 0, 0

        left_pt = [int(p1[0] + nx * lane_width), int(p1[1] + ny * lane_width)]
        right_pt = [int(p1[0] - nx * lane_width), int(p1[1] - ny * lane_width)]

        left_boundary.append(left_pt)
        right_boundary.append(right_pt)

    p_last = waypoints[-1]
    left_pt = [int(p_last[0] + nx * lane_width), int(p_last[1] + ny * lane_width)]
    right_pt = [int(p_last[0] - nx * lane_width), int(p_last[1] - ny * lane_width)]
    left_boundary.append(left_pt)
    right_boundary.append(right_pt)

    polygon = left_boundary + right_boundary[::-1]
    polygon = np.array([polygon], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return mask

def generate_perspective_lane_mask(img_shape, pts, base_width_bottom=250, base_width_top=20):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    left_bound = []
    right_bound = []

    n_pts = len(pts)
    for i in range(n_pts - 1):
        p1 = pts[i]
        p2 = pts[i+1]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            nx = -dy / length
            ny = dx / length
        else:
            nx, ny = 0, 0

        # Interpolate width based on Y position mapping to [0,1]
        t_w = i / max(1, n_pts - 1)
        # Width follows perspective depth (1/z) approximation
        current_width = base_width_bottom * (1 - t_w**0.5) + base_width_top * (t_w**0.5)

        left_bound.append([int(p1[0] + nx * current_width), int(p1[1] + ny * current_width)])
        right_bound.append([int(p1[0] - nx * current_width), int(p1[1] - ny * current_width)])

    # Last point
    p_last = pts[-1]
    t_w = 1.0
    current_width = base_width_top
    left_bound.append([int(p_last[0] + nx * current_width), int(p_last[1] + ny * current_width)])
    right_bound.append([int(p_last[0] - nx * current_width), int(p_last[1] - ny * current_width)])

    poly = left_bound + right_bound[::-1]
    cv2.fillPoly(mask, np.array([poly], dtype=np.int32), 255)
    return mask

def main():
    video_path = "/home/quynhthu/Documents/AI-project/YOLO/video1.mp4"
    if not Path(video_path).exists():
        print(f"Video {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to read frame.")
        return

    H, W = frame.shape[:2]

    waypoints = []
    # Start exactly at the bottom center (ego vehicle hood)
    start_x, start_y = W // 2, H

    # 3D to 2D perspective approximation
    # Assuming the camera is mounted forming a vanishing point roughly at H*0.4
    vanishing_y = int(H * 0.4)
    max_dist = start_y - vanishing_y

    for t in np.linspace(0, 1, 10):
        # Y goes from bottom of screen (H) towards vanishing point
        # Perspective makes distances compress as t approaches 1
        y = start_y - int(max_dist * (t ** 0.8))

        # X curves left, but the curve effect should be more pronounced closer to the bottom
        # and taper off near the vanishing point due to perspective.
        curve_factor = 250 * np.sin(t * np.pi / 2)
        x = start_x - int(curve_factor)

        waypoints.append([x, y])

    waypoints = np.array(waypoints)

    # Perspective Lane Width: wider at bottom, narrower at top
    # We'll adjust the mask generator to take varying widths

    lane_mask = generate_perspective_lane_mask(frame.shape, waypoints)

    color_mask = np.zeros_like(frame)
    color_mask[lane_mask == 255] = [0, 255, 0]

    alpha = 0.45
    valid_mask = lane_mask == 255
    blended = frame.copy()
    blended[valid_mask] = cv2.addWeighted(frame[valid_mask], 1 - alpha, color_mask[valid_mask], alpha, 0)

    # Center trajectory
    for i in range(len(waypoints) - 1):
        pt1 = tuple(waypoints[i])
        pt2 = tuple(waypoints[i+1])
        cv2.line(blended, pt1, pt2, (0, 0, 255), 4) # Red center line
        cv2.circle(blended, pt1, 6, (255, 0, 0), -1)

    out_path = "/home/quynhthu/Documents/AI-project/e2e/lane_mask_overlay.png"
    cv2.imwrite(out_path, blended)
    print(f"Saved perspective overlay visualization to {out_path}")

if __name__ == "__main__":
    main()
