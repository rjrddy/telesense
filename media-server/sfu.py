"""Small SFU built on aiortc.

One RTCPeerConnection per (room, peer). When a peer publishes a track, we
subscribe it via MediaRelay and add the subscription to every other peer's
PC, then nudge those peers to renegotiate via the signaling bus.

Renegotiation trigger: after we mutate a peer's PC, we send that peer a
`renegotiate` message through the Node signaling server. The client re-offers
via POST /offer, and this module reuses the existing PC to answer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

from media_pipeline import consume_audio, consume_video, subscribe
from model_interface import ModelInterface, get_model

log = logging.getLogger("sfu")


@dataclass
class Peer:
    peer_id: str
    room_id: str
    pc: RTCPeerConnection
    published: dict[str, MediaStreamTrack] = field(default_factory=dict)  # kind -> our subscriber-source track
    forwarded_from: set[str] = field(default_factory=set)                  # peer_ids whose tracks we've added to this PC
    consumer_tasks: list[asyncio.Task] = field(default_factory=list)


class Sfu:
    def __init__(self, notify_peer):
        # notify_peer(peer_id, payload) -> awaitable; sends a message through
        # the Node signaling bus to the given peer's WebSocket.
        self._notify_peer = notify_peer
        self._rooms: dict[str, dict[str, Peer]] = {}
        self._lock = asyncio.Lock()
        self._model = get_model()

    def _room(self, room_id: str) -> dict[str, Peer]:
        return self._rooms.setdefault(room_id, {})

    async def handle_offer(self, room_id: str, peer_id: str, sdp: str, sdp_type: str) -> dict:
        async with self._lock:
            room = self._room(room_id)
            peer = room.get(peer_id)
            if peer is None:
                peer = await self._create_peer(room_id, peer_id)
                room[peer_id] = peer

            # Make sure this peer is currently receiving every other peer's tracks.
            to_notify: set[str] = set()
            for other in room.values():
                if other.peer_id == peer_id:
                    continue
                for kind, track in other.published.items():
                    if other.peer_id in peer.forwarded_from:
                        continue
                    peer.pc.addTrack(subscribe(track))
                if other.peer_id not in peer.forwarded_from and other.published:
                    peer.forwarded_from.add(other.peer_id)

            await peer.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
            answer = await peer.pc.createAnswer()
            await peer.pc.setLocalDescription(answer)

            # If this peer now has published tracks, make sure every other peer
            # is forwarding them. New forwardings require the recipient to
            # renegotiate.
            for other in room.values():
                if other.peer_id == peer_id:
                    continue
                added_any = False
                for kind, track in peer.published.items():
                    if peer_id in other.forwarded_from:
                        continue
                    other.pc.addTrack(subscribe(track))
                    added_any = True
                if added_any or (peer.published and peer_id not in other.forwarded_from):
                    other.forwarded_from.add(peer_id)
                    to_notify.add(other.peer_id)

        # Fire renegotiation nudges outside the lock.
        for pid in to_notify:
            try:
                await self._notify_peer(pid, {"kind": "renegotiate"})
            except Exception:
                log.exception("failed to notify peer=%s for renegotiate", pid)

        return {"sdp": peer.pc.localDescription.sdp, "type": peer.pc.localDescription.type}

    async def _create_peer(self, room_id: str, peer_id: str) -> Peer:
        pc = RTCPeerConnection()
        peer = Peer(peer_id=peer_id, room_id=room_id, pc=pc)

        @pc.on("connectionstatechange")
        async def _on_state():
            log.info("peer=%s connectionState=%s", peer_id, pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await self.remove_peer(room_id, peer_id)

        @pc.on("track")
        def _on_track(track: MediaStreamTrack):
            log.info("peer=%s published %s track", peer_id, track.kind)
            # Keep a relay-subscribable source so other peers can consume it.
            source = subscribe(track)
            peer.published[track.kind] = source

            # Spin up the model consumer (uses a fresh subscription so it
            # doesn't compete with SFU forwarding).
            model_source = subscribe(track)
            if track.kind == "audio":
                task = asyncio.create_task(consume_audio(peer_id, model_source, self._model))
            elif track.kind == "video":
                task = asyncio.create_task(consume_video(peer_id, model_source, self._model))
            else:
                return
            peer.consumer_tasks.append(task)

            # Add this new track to every other peer already in the room.
            async def push_to_others():
                async with self._lock:
                    room = self._room(room_id)
                    notify_ids = []
                    for other in room.values():
                        if other.peer_id == peer_id:
                            continue
                        other.pc.addTrack(subscribe(track))
                        other.forwarded_from.add(peer_id)
                        notify_ids.append(other.peer_id)
                for pid in notify_ids:
                    try:
                        await self._notify_peer(pid, {"kind": "renegotiate"})
                    except Exception:
                        log.exception("failed notify peer=%s", pid)

            asyncio.create_task(push_to_others())

        return peer

    async def remove_peer(self, room_id: str, peer_id: str) -> None:
        async with self._lock:
            room = self._rooms.get(room_id, {})
            peer = room.pop(peer_id, None)
            if not peer:
                return
            for t in peer.consumer_tasks:
                t.cancel()
            try:
                await peer.pc.close()
            except Exception:
                pass
            log.info("peer=%s removed from room=%s", peer_id, room_id)

    async def shutdown(self) -> None:
        for room_id, room in list(self._rooms.items()):
            for peer_id in list(room.keys()):
                await self.remove_peer(room_id, peer_id)
