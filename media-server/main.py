"""Entry point for the Python media/ML worker.

Runs an aiohttp server that:
- accepts SDP offers at POST /offer
- opens a persistent WebSocket to the Node signaling server as `sfu-server`
  so it can (a) push `renegotiate` prompts back to specific peers and
  (b) receive chat messages to feed to the text model pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import aiohttp
import aiohttp_cors
from aiohttp import web

from sfu import Sfu
from model_interface import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("main")

SIGNALING_WS = os.environ.get("SIGNALING_WS", "ws://localhost:8081/ws?peer=sfu-server&room=default")
HTTP_PORT = int(os.environ.get("MEDIA_PORT", "8080"))


class SignalingBridge:
    """Long-lived WS client to the Node signaling server."""

    def __init__(self, url: str):
        self.url = url
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._task: asyncio.Task | None = None
        self._on_message = None  # set by caller

    def on_message(self, cb) -> None:
        self._on_message = cb

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        backoff = 1.0
        while True:
            try:
                assert self._session is not None
                async with self._session.ws_connect(self.url) as ws:
                    self._ws = ws
                    log.info("connected to signaling %s", self.url)
                    backoff = 1.0
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                            except json.JSONDecodeError:
                                continue
                            if self._on_message:
                                try:
                                    await self._on_message(data)
                                except Exception:
                                    log.exception("signaling handler failed")
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except Exception as e:
                log.warning("signaling connection error: %s", e)
            self._ws = None
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    async def send_to(self, peer_id: str, payload: dict[str, Any]) -> None:
        if not self._ws:
            log.warning("signaling not connected; dropping message to=%s", peer_id)
            return
        await self._ws.send_json({"type": "to", "to": peer_id, "payload": payload})

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._session:
            await self._session.close()


def create_app() -> web.Application:
    app = web.Application()

    bridge = SignalingBridge(SIGNALING_WS)
    sfu = Sfu(notify_peer=bridge.send_to)
    model = get_model()

    async def on_signal_message(data: dict[str, Any]) -> None:
        # Node forwards chat as {type: "chat", from, text, room}
        if data.get("type") == "chat":
            peer_id = data.get("from", "unknown")
            text = data.get("text", "")
            await model.feed_text(peer_id, text, source="chat")

    bridge.on_message(on_signal_message)

    async def offer_handler(request: web.Request) -> web.Response:
        body = await request.json()
        room_id = body.get("room", "default")
        peer_id = body["peer"]
        sdp = body["sdp"]
        sdp_type = body["type"]
        answer = await sfu.handle_offer(room_id, peer_id, sdp, sdp_type)
        return web.json_response(answer)

    async def leave_handler(request: web.Request) -> web.Response:
        body = await request.json()
        await sfu.remove_peer(body.get("room", "default"), body["peer"])
        return web.json_response({"ok": True})

    async def health(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def on_startup(_: web.Application) -> None:
        await bridge.start()

    async def on_cleanup(_: web.Application) -> None:
        await sfu.shutdown()
        await bridge.stop()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # CORS so the Vite dev server (localhost:5173) can call us.
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=False, expose_headers="*", allow_headers="*", allow_methods="*",
        ),
    })
    cors.add(app.router.add_post("/offer", offer_handler))
    cors.add(app.router.add_post("/leave", leave_handler))
    cors.add(app.router.add_get("/health", health))

    return app


def main() -> None:
    web.run_app(create_app(), host="0.0.0.0", port=HTTP_PORT)


if __name__ == "__main__":
    main()
