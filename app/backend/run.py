"""Run script for Telesense backend server."""

import sys
from pathlib import Path

# Make the repo root (app) and src/ (telesense_ml) importable
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

if __name__ == "__main__":
    import uvicorn

    # Localhost only: the API has no authentication. reload needs an import string, not the app object
    uvicorn.run("app.backend.main:app", host="127.0.0.1", port=8000, reload=True)
