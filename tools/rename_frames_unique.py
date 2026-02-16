from pathlib import Path
import argparse

def rename_frames_unique(images_dir: Path):
    """
    Renames frames in subdirectories to ensure global uniqueness by prepending the directory name.
    """
    if not images_dir.exists():
        print(f"Directory {images_dir} does not exist.")
        return

    # Iterate over subdirectories (assuming one per video)
    for video_dir in images_dir.iterdir():
        if not video_dir.is_dir():
            continue

        video_name = video_dir.name
        print(f"Processing directory: {video_name}")

        renamed_count = 0
        skipped_count = 0
        
        # Iterate over files in the subdirectory
        for file_path in video_dir.iterdir():
            if not file_path.is_file():
                continue
            
            # Check if file is an image (simple check)
            if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

            # Check if already prefixed
            if file_path.name.startswith(f"{video_name}_"):
                skipped_count += 1
                continue
            
            # Construct new name: FolderName_OriginalName
            new_name = f"{video_name}_{file_path.name}"
            new_path = video_dir / new_name
            
            try:
                file_path.rename(new_path)
                renamed_count += 1
            except OSError as e:
                print(f"Error renaming {file_path.name}: {e}")

        print(f"  Renamed: {renamed_count}, Skipped (already unique): {skipped_count}")

def main():
    parser = argparse.ArgumentParser(description="Rename frames in subdirectories to be unique.")
    parser.add_argument("--dir", type=str, default="data/images", help="Directory containing video subfolders")
    
    args = parser.parse_args()
    images_dir = Path(args.dir)
    
    rename_frames_unique(images_dir)

if __name__ == "__main__":
    main()
