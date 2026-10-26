"""FastAPI application for Telesense video analysis."""

import os
import secrets
from tempfile import SpooledTemporaryFile
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, QueryParams

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


# ----- Config via env (all optional) -----
# Comma-separated list of allowed origins; "*" allows any (dev default).
_CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]

# If set, every /api/* request must present this value as the X-Telesense-Key
# header (HTTP) or ?key=<value> query param (WebSocket). Leave unset for local dev.
_DEMO_KEY = os.environ.get("TELESENSE_DEMO_KEY", "").strip()

# Limit the entire multipart request, including uploads without Content-Length.
_MAX_UPLOAD_MB = int(os.environ.get("TELESENSE_MAX_UPLOAD_MB", "100"))
if _MAX_UPLOAD_MB <= 0:
    raise ValueError("TELESENSE_MAX_UPLOAD_MB must be positive")
_MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024


class SizeLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        response = JSONResponse(status_code=413, content={
            "detail": f"Upload too large (max {_MAX_UPLOAD_MB} MB including multipart overhead)"
        })
        length = Headers(scope=scope).get("content-length")
        if length and length.isdigit() and int(length) > _MAX_UPLOAD_BYTES:
            return await response(scope, receive, send)
        # Bound memory, and reject before multipart parsing or route side effects.
        with SpooledTemporaryFile(max_size=1024 * 1024) as body:
            total = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    return await response(scope, receive, send)
                body.write(chunk)
                if not message.get("more_body", False):
                    break
            body.seek(0)
            complete = False

            async def replay():
                nonlocal complete
                if complete:
                    return await receive()
                chunk = body.read(64 * 1024)
                complete = body.tell() == total
                return {"type": "http.request", "body": chunk, "more_body": not complete}

            await self.app(scope, replay, send)


class DemoKeyMiddleware:
    """Gate HTTP and WebSocket API scopes before routing."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        path = scope.get("path", "")
        if not _DEMO_KEY or kind not in {"http", "websocket"} or not (
            path == "/api" or path.startswith("/api/")
        ):
            return await self.app(scope, receive, send)
        presented = Headers(scope=scope).get("x-telesense-key", "")
        # Media elements and WebSocket clients cannot set custom headers.
        if kind == "websocket" or (scope.get("method") == "GET" and path.startswith("/api/audio/")):
            presented = presented or QueryParams(scope.get("query_string", b"").decode()).get("key", "")
        if not secrets.compare_digest(presented.encode(), _DEMO_KEY.encode()):
            if kind == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            return await JSONResponse(status_code=401, content={
                "detail": "Invalid or missing API key"
            })(scope, receive, send)
        return await self.app(scope, receive, send)


app = FastAPI(
    title="Telesense API",
    description="Research behavioral analysis demo. Not for clinical use.",
    version="1.0.0",
)

app.add_middleware(SizeLimitMiddleware)
app.add_middleware(DemoKeyMiddleware)
# Last added is outermost: preflights and error responses need CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*", "X-Telesense-Key"],
)

app.include_router(router, prefix="/api", tags=["api"])


@app.get("/")
async def root():
    return {
        "message": "Telesense API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "analyze": "/api/analyze/{video_id}",
            "status": "/api/status/{job_id}",
            "results": "/api/results/{job_id}",
            "websocket": "/api/ws/{job_id}",
        },
        "auth": "X-Telesense-Key header (or ?key=... for WS) required" if _DEMO_KEY else "open (no key set)",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
