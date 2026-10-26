export interface WebRtcClientOptions {
  mediaServerUrl: string; // e.g. http://localhost:8080
  room: string;
  peerId: string;
  onRemoteStream: (peerId: string, stream: MediaStream) => void;
  onLocalStream: (stream: MediaStream) => void;
  onState?: (state: RTCPeerConnectionState) => void;
}

export class WebRtcClient {
  private pc?: RTCPeerConnection;
  private local?: MediaStream;
  private negotiating = false;
  private pendingRenegotiate = false;
  private remoteStreams = new Map<string, MediaStream>();

  constructor(private readonly opts: WebRtcClientOptions) {}

  async start(): Promise<void> {
    const pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });
    this.pc = pc;

    pc.onconnectionstatechange = () => {
      this.opts.onState?.(pc.connectionState);
    };

    pc.ontrack = (ev) => {
      // The SFU may add many remote tracks (one per remote peer). aiortc adds
      // each track with a fresh MSID/stream, so we key remote streams by the
      // first stream id we see for that track.
      const [stream] = ev.streams;
      const key = stream ? stream.id : ev.track.id;
      let s = this.remoteStreams.get(key);
      if (!s) {
        s = new MediaStream();
        this.remoteStreams.set(key, s);
      }
      s.addTrack(ev.track);
      this.opts.onRemoteStream(key, s);
    };

    this.local = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
    for (const t of this.local.getTracks()) pc.addTrack(t, this.local);
    this.opts.onLocalStream(this.local);

    await this.negotiate();
  }

  async renegotiate(): Promise<void> {
    if (!this.pc) return;
    if (this.negotiating) {
      this.pendingRenegotiate = true;
      return;
    }
    await this.negotiate();
    if (this.pendingRenegotiate) {
      this.pendingRenegotiate = false;
      await this.renegotiate();
    }
  }

  private async negotiate(): Promise<void> {
    const pc = this.pc;
    if (!pc) return;
    this.negotiating = true;
    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const res = await fetch(`${this.opts.mediaServerUrl}/offer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          room: this.opts.room,
          peer: this.opts.peerId,
          sdp: offer.sdp,
          type: offer.type,
        }),
      });
      if (!res.ok) throw new Error(`offer failed: ${res.status}`);
      const answer = (await res.json()) as RTCSessionDescriptionInit;
      await pc.setRemoteDescription(answer);
    } finally {
      this.negotiating = false;
    }
  }

  async stop(): Promise<void> {
    try {
      await fetch(`${this.opts.mediaServerUrl}/leave`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room: this.opts.room, peer: this.opts.peerId }),
        keepalive: true,
      });
    } catch { /* ignore */ }
    this.local?.getTracks().forEach((t) => t.stop());
    this.pc?.close();
    this.pc = undefined;
  }
}
