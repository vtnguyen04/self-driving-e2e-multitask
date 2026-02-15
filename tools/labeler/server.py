import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json
import glob
import os

from fastapi.responses import FileResponse

# Lifespan (Startup/Shutdown)
from contextlib import asynccontextmanager
from migrate import sync_images

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure DB is synced with raw images
    try:
        conn = get_db_connection()
        sync_images(conn)
        conn.close()
    except Exception as e:
        print(f"Startup Sync Failed: {e}")
    yield
    # Shutdown

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
# ../../data relative to tools/labeler/server.py -> e2e/data
DATA_DIR = BASE_DIR.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

class LabelData(BaseModel):
    image_path: str
    command: int
    # Speed/Steer Removed
    waypoints: list[list[float]] = []
    bboxes: list[list[float]] = [] # [x, y, w, h]
    categories: list[int] = [] # class ids
    control_points: list[list[float]] = [] # Optional: [P0, P1, P2, P3] for Bezier

from db_utils import get_db_connection

@app.get("/api/images")
async def list_images():
    """List all images with labeled status from DB."""
    # Note: Auto-sync happens on startup now.
    conn = get_db_connection()
    c = conn.cursor()
    # Order by ID to keep duplicates next to originals if inserted later?
    # Or alphanumeric? Alphanumeric is better.
    c.execute("SELECT image_name, is_labeled FROM samples ORDER BY image_name")
    rows = c.fetchall()
    conn.close()

    # Return list of {name: str, labeled: bool}
    return [{"name": r["image_name"], "labeled": bool(r["is_labeled"])} for r in rows]

@app.get("/api/image/{filename}")
async def get_image(filename: str):
    """Serve image file."""
    # DB lookup to get REAL path (since filename might be a virtual duplicate)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT image_path FROM samples WHERE image_name = ?", (filename,))
    row = c.fetchone()
    conn.close()

    if not row:
         # Fallback for old behavior (direct file)
         file_path = RAW_DIR / filename
    else:
         file_path = Path(row['image_path'])

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

@app.post("/api/duplicate/{filename}")
async def duplicate_sample(filename: str):
    """Duplicate a sample in DB (same image, new entry)."""
    conn = get_db_connection()
    c = conn.cursor()

    # 1. Get original info
    c.execute("SELECT image_path FROM samples WHERE image_name = ?", (filename,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Original image not found")

    original_path = row['image_path']
    base_name = Path(filename).stem
    suffix = Path(filename).suffix

    # 2. Find next available name
    # Simplistic: <name>_dup<N><suffix>
    import re
    # Check if already a duplicate
    match = re.search(r'_dup(\d+)$', base_name)
    if match:
        base_name = base_name[:match.start()] # strip existing dup suffix

    # Count existing duplicates
    c.execute("SELECT image_name FROM samples WHERE image_name LIKE ?", (f"{base_name}_dup%",))
    existing_dups = [r['image_name'] for r in c.fetchall()]

    cnt = 1
    while True:
        new_name = f"{base_name}_dup{cnt}{suffix}"
        if new_name not in existing_dups and new_name != filename: # logic check
             # Also check uniqueness in DB constraint just in case
             try:
                 c.execute(
                    "INSERT INTO samples (image_name, image_path, is_labeled, data) VALUES (?, ?, 0, NULL)",
                    (new_name, original_path)
                 )
                 conn.commit()
                 conn.close()
                 return {"status": "success", "new_name": new_name}
             except sqlite3.IntegrityError:
                 pass # Name collision, try next
        cnt += 1

    conn.close()
    raise HTTPException(status_code=500, detail="Failed to generate duplicate name")

@app.post("/api/duplicate/{filename}")
async def duplicate_sample(filename: str):
    """Duplicate a sample in DB (same image, new entry)."""
    conn = get_db_connection()
    c = conn.cursor()

    # 1. Get original info
    c.execute("SELECT image_path FROM samples WHERE image_name = ?", (filename,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Original image not found")

    original_path = row['image_path']
    base_name = Path(filename).stem
    suffix = Path(filename).suffix

    # 2. Find next available name
    # Simplistic: <name>_dup<N><suffix>
    import re
    # Check if already a duplicate
    match = re.search(r'_dup(\d+)$', base_name)
    if match:
        base_name = base_name[:match.start()] # strip existing dup suffix

    # Count existing duplicates
    c.execute("SELECT image_name FROM samples WHERE image_name LIKE ?", (f"{base_name}_dup%",))
    existing_dups = [r['image_name'] for r in c.fetchall()]

    cnt = 1
    while True:
        new_name = f"{base_name}_dup{cnt}{suffix}"
        if new_name not in existing_dups and new_name != filename: # logic check
             # Also check uniqueness in DB constraint just in case
             try:
                 c.execute(
                    "INSERT INTO samples (image_name, image_path, is_labeled, data) VALUES (?, ?, 0, NULL)",
                    (new_name, original_path)
                 )
                 conn.commit()
                 conn.close()
                 return {"status": "success", "new_name": new_name}
             except sqlite3.IntegrityError:
                 pass # Name collision, try next
        cnt += 1

    conn.close()
    raise HTTPException(status_code=500, detail="Failed to generate duplicate name")


@app.post("/api/save")
async def save_label(data: LabelData):
    """Save label data to SQLite."""
    filename = Path(data.image_path).name
    # Wait, data.image_path comes from UI. If currentFile is "img_dup1.jpg",
    # the UI might still be sending the physical path in `image_path` field?
    # NO. The UI sends what it thinks is the path.
    # ACTUALLY: The UI logic `image_path: /home/.../${currentFile}` constructs a fake path if currentFile is virtual.
    # We should trust `currentFile` passed in URL or rely on lookup.
    # But `LabelData` has `image_path`.
    # Quick Fix: In `save_label`, we should match by `filename`.
    # But `filename` is derived from `data.image_path`.
    # If the UI sends `.../img_dup1.jpg`, `filename` becomes `img_dup1.jpg`.
    # This works IF the UI sends the virtual filename in the path string.

    filename = Path(data.image_path).name

    conn = get_db_connection()
    c = conn.cursor()

    # Store the entire LabelData model as JSON
    json_str = json.dumps(data.model_dump())

    try:
        # We update by image_name (which is unique)
        c.execute(
            "UPDATE samples SET data = ?, is_labeled = 1, updated_at = CURRENT_TIMESTAMP WHERE image_name = ?",
            (json_str, filename)
        )
        if c.rowcount == 0:
             conn.close()
             raise HTTPException(status_code=404, detail=f"Image {filename} not found in DB")
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"DB Update Failed: {str(e)}")

    conn.close()

    # 2. Backup to JSON File (Virtual Name)
    try:
        json_path = PROCESSED_DIR / (Path(filename).stem + ".json")
        with open(json_path, "w") as f:
            f.write(json.dumps(data.model_dump(), indent=2))
    except Exception as e:
        print(f"Warning: Failed to save JSON backup: {e}")

    return {"status": "success", "image": filename}

@app.post("/api/reset/{filename}")
async def reset_label(filename: str):
    """Reset label (Unlabel) for a sample."""
    conn = get_db_connection()
    c = conn.cursor()
    try:
        # 1. Update DB
        c.execute(
            "UPDATE samples SET data = NULL, is_labeled = 0, updated_at = CURRENT_TIMESTAMP WHERE image_name = ?",
            (filename,)
        )
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))
    conn.close()

    # 2. Delete JSON Backup
    try:
        json_path = PROCESSED_DIR / (Path(filename).stem + ".json")
        if json_path.exists():
            os.remove(json_path)
    except Exception as e:
        print(f"Warning: Failed to delete JSON backup: {e}")

    return {"status": "success", "image": filename}

@app.get("/api/label/{filename}")
async def get_label(filename: str):
    """Load label from SQLite."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT data FROM samples WHERE image_name = ? AND is_labeled = 1", (filename,))
    row = c.fetchone()
    conn.close()

    if row and row["data"]:
        return json.loads(row["data"])
    return None

@app.delete("/api/image/{filename}")
async def delete_image(filename: str):
    """Delete an image and its label. Deletes physical file only if no other references exist."""
    conn = get_db_connection()
    c = conn.cursor()

    # 1. Get info to decide on physical deletion
    c.execute("SELECT image_path FROM samples WHERE image_name = ?", (filename,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = row['image_path']

    # 2. Delete from DB
    try:
        c.execute("DELETE FROM samples WHERE image_name = ?", (filename,))
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Check if any other sample references this physical file
    c.execute("SELECT count(*) as cnt FROM samples WHERE image_path = ?", (image_path,))
    ref_count = c.fetchone()['cnt']
    conn.close()

    # 4. Delete Physical File if orphaned
    deleted_file = False
    if ref_count == 0:
        try:
            p = Path(image_path)
            if p.exists():
                os.remove(p)
                deleted_file = True
        except Exception as e:
            print(f"Error deleting file {image_path}: {e}")

    # 5. Delete Label JSON (Always, since it is tied to image_name)
    try:
        json_path = PROCESSED_DIR / (Path(filename).stem + ".json")
        if json_path.exists():
            os.remove(json_path)
    except Exception as e:
         print(f"Warning: Failed to delete JSON backup: {e}")

    return {"status": "success", "deleted_file": deleted_file}

@app.delete("/api/images/bulk")
async def delete_images_bulk(filenames: list[str] = Body(...)):
    """Delete multiple images at once."""
    deleted_count = 0
    deleted_files = 0
    errors = []

    conn = get_db_connection()
    c = conn.cursor()

    for filename in filenames:
        try:
            # 1. Get info
            c.execute("SELECT image_path FROM samples WHERE image_name = ?", (filename,))
            row = c.fetchone()
            if not row:
                errors.append(f"{filename}: not found")
                continue

            image_path = row['image_path']

            # 2. Delete from DB
            c.execute("DELETE FROM samples WHERE image_name = ?", (filename,))
            deleted_count += 1

            # 3. Check if orphaned
            c.execute("SELECT count(*) as cnt FROM samples WHERE image_path = ?", (image_path,))
            ref_count = c.fetchone()['cnt']

            # 4. Delete Physical File if orphaned
            if ref_count == 0:
                try:
                    p = Path(image_path)
                    if p.exists():
                        os.remove(p)
                        deleted_files += 1
                except Exception as e:
                    errors.append(f"{filename}: file delete error - {e}")

            # 5. Delete Label JSON
            try:
                json_path = PROCESSED_DIR / (Path(filename).stem + ".json")
                if json_path.exists():
                    os.remove(json_path)
            except:
                pass

        except Exception as e:
            errors.append(f"{filename}: {str(e)}")

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "deleted_entries": deleted_count,
        "deleted_files": deleted_files,
        "errors": errors
    }

# Mount static files (Frontend)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")

@app.get("/")
async def read_index():
    return FileResponse(BASE_DIR / "static" / "index.html")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
