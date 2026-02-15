import shutil
from pathlib import Path
import os

def merge_folders(base_dir: Path, target_dir_name: str = "merged_frames"):
    """
    Moves all files from subdirectories starting with 'Video_' into a single target directory.
    """
    target_dir = base_dir / target_dir_name
    target_dir.mkdir(exist_ok=True)
    
    print(f"Target directory: {target_dir}")

    # List only directories starting with Video_
    subdirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("Video_")]

    if not subdirs:
        print("No 'Video_' directories found to merge.")
        return

    total_moved = 0

    for subdir in subdirs:
        print(f"Processing folder: {subdir.name}")
        files = [f for f in subdir.iterdir() if f.is_file()]
        
        for file_path in files:
            # New path in the target directory
            # Note: We assume files are already uniquely named from the previous step
            # If not, we should handle collisions here too, just in case.
            dest_path = target_dir / file_path.name
            
            # Safety check for duplicates
            if dest_path.exists():
                # If collision happens, append a counter
                stem = dest_path.stem
                suffix = dest_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = target_dir / f"{stem}_dup{counter}{suffix}"
                    counter += 1
                print(f"  Warning: Collision resolved -> {dest_path.name}")
            
            shutil.move(str(file_path), str(dest_path))
            total_moved += 1
        
        # Remove the now empty directory
        try:
            subdir.rmdir()
            print(f"  Removed empty folder: {subdir.name}")
        except OSError as e:
            print(f"  Could not remove folder {subdir.name} (not empty?): {e}")

    print(f"Done. Moved {total_moved} files to {target_dir}")

if __name__ == "__main__":
    base_images_path = Path("data/images")
    merge_folders(base_images_path)
