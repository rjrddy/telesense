import { useEffect, useMemo, useRef, useState } from "react";
import { SignalingClient, SignalMessage } from "./lib/signaling";
import { WebRtcClient } from "./lib/webrtc";
import { VideoTile } from "./components/VideoTile";
import { Chat, ChatMessage } from "./components/Chat";

interface Score {
  peer: string;
  label: string;
  confidence: number;
  probs: number[];
}

const SIGNAL_URL = import.meta.env.VITE_SIGNAL_URL ?? "ws://localhost:8081/ws";
const MEDIA_URL = import.meta.env.VITE_MEDIA_URL ?? "http://localhost:8080";
const ROOM = "default";

function makePeerId(): string {
  return "p_" + Math.random().toString(36).slice(2, 10);
}

export function App() {
  const peerId = useMemo(makePeerId, []);
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remotes, setRemotes] = useState<Map<string, MediaStream>>(new Map());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [roster, setRoster] = useState<string[]>([]);
  const [connState, setConnState] = useState<RTCPeerConnectionState>("new");
  const [scores, setScores] = useState<Map<string, Score>>(new Map());
  const rtcRef = useRef<WebRtcClient | null>(null);
  const sigRef = useRef<SignalingClient | null>(null);

  useEffect(() => {
    const sig = new SignalingClient(SIGNAL_URL, peerId, ROOM);
    sigRef.current = sig;

    const rtc = new WebRtcClient({
      mediaServerUrl: MEDIA_URL,
      room: ROOM,
      peerId,
      onLocalStream: setLocalStream,
      onRemoteStream: (key, stream) => {
        setRemotes((prev) => {
          const next = new Map(prev);
          next.set(key, stream);
          return next;
        });
      },
      onState: setConnState,
    });
    rtcRef.current = rtc;

    const off = sig.on((msg: SignalMessage) => {
      switch (msg.type) {
        case "hello":
        case "roster":
          setRoster(msg.peers);
          break;
        case "chat":
          setMessages((prev) => [...prev, { from: msg.from, text: msg.text }]);
          break;
        case "from": {
          const p = msg.payload as { kind?: string } | null;
          if (p?.kind === "renegotiate") {
            void rtc.renegotiate();
          } else if (p?.kind === "score") {
            const s = p as unknown as Score;
            setScores((prev) => {
              const next = new Map(prev);
              next.set(s.peer, s);
              return next;
            });
          }
          break;
        }
      }
    });

    sig.connect();
    void rtc.start().catch((err) => console.error("rtc.start failed", err));

    const beforeUnload = () => { void rtc.stop(); };
    window.addEventListener("beforeunload", beforeUnload);

    return () => {
      off();
      window.removeEventListener("beforeunload", beforeUnload);
      void rtc.stop();
      sig.close();
    };
  }, [peerId]);

  const sendChat = (text: string) => sigRef.current?.sendChat(text);

  const remoteTiles = [...remotes.entries()];

  return (
    <div className="app">
      <div>
        <h1>Telesense — room: {ROOM} · you: {peerId.slice(0, 8)}</h1>
        <div className="videos">
          <VideoTile stream={localStream} label={`you (${peerId.slice(0, 6)})`} muted />
          {remoteTiles.length === 0 ? (
            <div className="video-tile">
              <div className="label">waiting for peer…</div>
            </div>
          ) : (
            remoteTiles.map(([key, s]) => (
              <VideoTile key={key} stream={s} label={`peer ${key.slice(0, 6)}`} />
            ))
          )}
        </div>
      </div>
      <div className="sidebar">
        <div className="status">
          <div>webrtc: {connState}</div>
          <div>peers in room: {roster.length} ({roster.map((p) => p.slice(0, 6)).join(", ") || "—"})</div>
        </div>
        <div className="scores">
          <div className="scores-title">Model scores (7 classes)</div>
          {scores.size === 0 ? (
            <div className="scores-empty">no scores yet…</div>
          ) : (
            [...scores.values()].map((s) => (
              <div key={s.peer} className="score-row">
                <div className="score-head">
                  <span>{s.peer.slice(0, 6)}</span>
                  <span className="score-label">{s.label}</span>
                  <span className="score-conf">{(s.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="score-bars">
                  {s.probs.map((p, i) => (
                    <div key={i} className="score-bar" title={`class ${i}: ${(p * 100).toFixed(1)}%`}>
                      <div className="score-fill" style={{ height: `${p * 100}%` }} />
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
        <Chat messages={messages} onSend={sendChat} me={peerId} />
      </div>
    </div>
  );
}
