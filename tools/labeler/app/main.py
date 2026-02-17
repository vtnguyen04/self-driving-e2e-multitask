from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .routers import label_router, version_router, upload_router
from .core.config import Config
from pathlib import Path

app = FastAPI(title=Config.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. API Routes (High Priority)
app.include_router(label_router.router)
app.include_router(version_router.router)
app.include_router(upload_router.router)

# UI Configuration
ui_dist = Path(__file__).resolve().parent.parent / "ui" / "dist"

if ui_dist.exists():
    # 2. Specifically mount the assets directory for hashed JS/CSS
    # This ensures /assets/... requests are served efficiently
    app.mount("/assets", StaticFiles(directory=str(ui_dist / "assets")), name="assets")

    # 3. Serve individual files from the root of dist (like favicons, vite.svg)
    # But ONLY if they exist. Otherwise, we want them to fall through to the SPA handler or API.

    @app.get("/")
    async def serve_index():
        return FileResponse(ui_dist / "index.html")

    # Catch-all handler for SPA and other root-level files
    @app.get("/{full_path:path}")
    async def catch_all(full_path: str):
        # Prevent catching API or Assets incorrectly
        if full_path.startswith("api/") or full_path.startswith("assets/"):
            raise HTTPException(status_code=404)

        # Check if the file exists physically (e.g., /vite.svg)
        file_path = ui_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        # Otherwise, serve index.html for React Router to handle
        return FileResponse(ui_dist / "index.html")
else:
    @app.get("/")
    async def fallback():
        return {
            "status": "warning",
            "message": "NeuroPilot Labeler Pro API is running, but Frontend was not found.",
            "hint": "Run 'npm run build' in tools/labeler/ui to enable the Premium interface."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
