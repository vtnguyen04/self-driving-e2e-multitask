import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path: Path, output_dir: Path, sample_rate: int = 10):
    """
    Extracts frames from a video file.
    
    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        sample_rate: Save every Nth frame.
    """
    if not video_path.exists():
        print(f"Error: Video file {video_path} not found.")
        return

    # Create output directory for this specific video
    video_name = video_path.stem
    save_dir = output_dir / video_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {video_path.name} -> {save_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    saved_count = 0

    pbar = tqdm(total=total_frames, desc=f"Extracting {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            frame_filename = save_dir / f"{video_name}_frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1

        frame_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f"Finished. Saved {saved_count} frames to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos in data/video")
    parser.add_argument("--source", type=str, default="data/video", help="Source directory containing videos")
    parser.add_argument("--output", type=str, default="data/images", help="Output directory for frames")
    parser.add_argument("--rate", type=int, default=10, help="Save every Nth frame (default: 10)")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    if not source_dir.exists():
        print(f"Source directory {source_dir} does not exist.")
        return

    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [f for f in source_dir.iterdir() if f.suffix.lower() in video_extensions]

    if not videos:
        print(f"No video files found in {source_dir}")
        return

    print(f"Found {len(videos)} videos. Starting extraction...")
    
    for video in videos:
        extract_frames(video, output_dir, args.rate)

if __name__ == "__main__":
    main()
