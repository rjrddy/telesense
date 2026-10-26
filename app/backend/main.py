"""FastAPI application for Telesense video analysis."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

try:
    from .api.routes import router
except ImportError:
    # For direct execution, add parent to path
    import sys
    backend_dir = Path(__file__).parent
    parent_dir = backend_dir.parent.parent
    for path in (parent_dir, parent_dir / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from app.backend.api.routes import router

# Create FastAPI app
app = FastAPI(
    title="Telesense API",
    description="Psychiatric Evaluation Video Analysis System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["api"])


@app.get("/")
async def root():
    """
    Root endpoint.
    Input:
        None

    Output:
        API name, version and endpoint list
    """
    return {
        "message": "Telesense API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "analyze": "/api/analyze/{video_id}",
            "status": "/api/status/{job_id}",
            "results": "/api/results/{job_id}",
            "websocket": "/api/ws/{job_id}"
        }
    }


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Input:
        None

    Output:
        {'status': 'healthy'}
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    # Localhost only: the API has no authentication
    uvicorn.run(app, host="127.0.0.1", port=8000)
