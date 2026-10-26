export type SignalMessage =
  | { type: "hello"; peer: string; room: string; peers: string[] }
  | { type: "roster"; peers: string[]; room: string }
  | { type: "chat"; from: string; text: string; room: string }
  | { type: "from"; from: string; payload: unknown };

type Listener = (msg: SignalMessage) => void;

export class SignalingClient {
  private ws?: WebSocket;
  private listeners = new Set<Listener>();
  private reconnectDelay = 1000;

  constructor(
    private readonly url: string,
    readonly peerId: string,
    readonly room: string,
  ) {}

  connect(): void {
    const u = new URL(this.url);
    u.searchParams.set("peer", this.peerId);
    u.searchParams.set("room", this.room);
    const ws = new WebSocket(u.toString());
    this.ws = ws;

    ws.onopen = () => {
      this.reconnectDelay = 1000;
    };
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as SignalMessage;
        for (const l of this.listeners) l(data);
      } catch {
        // ignore
      }
    };
    ws.onclose = () => {
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, 10000);
    };
    ws.onerror = () => ws.close();
  }

  on(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  sendChat(text: string): void {
    this.ws?.send(JSON.stringify({ type: "chat", text }));
  }

  close(): void {
    this.ws?.close();
    this.ws = undefined;
  }
}
