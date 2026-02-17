import uvicorn
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))
sys.path.append(str(Path(__file__).parent.resolve()))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
