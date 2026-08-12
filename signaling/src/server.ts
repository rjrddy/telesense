import { WebSocketServer, WebSocket } from "ws";
import { createServer } from "http";
import { URL } from "url";

const PORT = Number(process.env.SIGNAL_PORT ?? 8081);

type PeerId = string;
type RoomId = string;

interface Client {
  peerId: PeerId;
  roomId: RoomId;
  ws: WebSocket;
}

const rooms = new Map<RoomId, Map<PeerId, Client>>();

function getRoom(roomId: RoomId): Map<PeerId, Client> {
  let room = rooms.get(roomId);
  if (!room) {
    room = new Map();
    rooms.set(roomId, room);
  }
  return room;
}

function humanPeers(room: Map<PeerId, Client>): PeerId[] {
  return [...room.keys()].filter((p) => p !== "sfu-server");
}

function broadcastRoster(roomId: RoomId): void {
  const room = getRoom(roomId);
  const peers = humanPeers(room);
  const msg = JSON.stringify({ type: "roster", peers, room: roomId });
  for (const c of room.values()) c.ws.send(msg);
}

function safeSend(ws: WebSocket, obj: unknown): void {
  if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj));
}

const http = createServer();
const wss = new WebSocketServer({ noServer: true });

http.on("upgrade", (req, socket, head) => {
  if (!req.url) {
    socket.destroy();
    return;
  }
  const url = new URL(req.url, `http://${req.headers.host}`);
  if (url.pathname !== "/ws") {
    socket.destroy();
    return;
  }
  const peerId = url.searchParams.get("peer");
  const roomId = url.searchParams.get("room") ?? "default";
  if (!peerId) {
    socket.destroy();
    return;
  }
  wss.handleUpgrade(req, socket, head, (ws) => {
    wss.emit("connection", ws, { peerId, roomId });
  });
});

wss.on("connection", (ws: WebSocket, ctx: { peerId: PeerId; roomId: RoomId }) => {
  const { peerId, roomId } = ctx;
  const room = getRoom(roomId);

  const existing = room.get(peerId);
  if (existing) {
    try { existing.ws.close(4000, "replaced"); } catch { /* ignore */ }
  }

  const client: Client = { peerId, roomId, ws };
  room.set(peerId, client);
  console.log(`[join] peer=${peerId} room=${roomId} (size=${room.size})`);

  safeSend(ws, { type: "hello", peer: peerId, room: roomId, peers: humanPeers(room) });
  broadcastRoster(roomId);

  ws.on("message", (raw) => {
    let msg: any;
    try { msg = JSON.parse(raw.toString()); } catch { return; }
    if (!msg || typeof msg !== "object") return;

    switch (msg.type) {
      case "chat": {
        if (typeof msg.text !== "string") return;
        const out = { type: "chat", from: peerId, text: msg.text, room: roomId };
        for (const c of room.values()) safeSend(c.ws, out);
        break;
      }
      case "to": {
        // Direct message to a specific peer (e.g. sfu-server -> browser renegotiate)
        const target = room.get(String(msg.to));
        if (!target) return;
        safeSend(target.ws, { type: "from", from: peerId, payload: msg.payload });
        break;
      }
      default:
        // ignore unknowns; keeps forward-compat cheap
        break;
    }
  });

  ws.on("close", () => {
    // Only remove if this ws is still the one registered (avoid removing the replacement)
    if (room.get(peerId)?.ws === ws) {
      room.delete(peerId);
      console.log(`[leave] peer=${peerId} room=${roomId} (size=${room.size})`);
      broadcastRoster(roomId);
    }
  });
});

http.listen(PORT, () => {
  console.log(`signaling server listening on :${PORT} (ws path /ws)`);
});
