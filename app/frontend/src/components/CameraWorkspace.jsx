import { useCallback, useEffect, useRef, useState } from "react";
import Icon from "./Icon";
import { loadCameraModel } from "../services/cameraModel";
import { time } from "./SessionPlayer";

const LABELS = {
  neutral: "Neutral-looking",
  happy: "Happy-looking",
  sad: "Sad-looking",
  angry: "Angry-looking",
  fearful: "Fear-like expression",
  disgusted: "Disgust-like expression",
  surprised: "Surprised-looking",
};
const ORDER = Object.keys(LABELS);
const COLORS = {
  neutral: "#8cafbd",
  happy: "#a9cf9a",
  sad: "#a2a6d7",
  angry: "#ce9691",
  fearful: "#bba0d3",
  disgusted: "#b8b484",
  surprised: "#d2bb85",
};
const messageFor = (error) => {
  if (["NotAllowedError", "PermissionDeniedError"].includes(error.name))
    return "Camera access was denied. Allow camera access in your browser’s site settings, then try again.";
  if (["NotFoundError", "DevicesNotFoundError"].includes(error.name))
    return "No camera was found. Connect a camera and try again.";
  if (["NotReadableError", "TrackStartError"].includes(error.name))
    return "The camera is busy or unavailable. Close other apps using it and try again.";
  return (
    error.message || "Camera evaluation could not start. Please try again."
  );
};

export default function CameraWorkspace() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const tokenRef = useRef(0);
  const timerRef = useRef(null);
  const historyRef = useRef([]);
  const startedRef = useRef(0);
  const mountedRef = useRef(true);
  const [phase, setPhase] = useState("idle");
  const [error, setError] = useState("");
  const [status, setStatus] = useState("Camera is off");
  const [scores, setScores] = useState(null);
  const [box, setBox] = useState(null);
  const [size, setSize] = useState({ width: 640, height: 480 });
  const [history, setHistory] = useState([]);
  const [elapsed, setElapsed] = useState(0);
  const [latency, setLatency] = useState(null);
  const [faceCount, setFaceCount] = useState(0);

  const release = useCallback(() => {
    tokenRef.current += 1;
    clearTimeout(timerRef.current);
    streamRef.current?.getTracks().forEach((track) => {
      track.onended = null;
      track.stop();
    });
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);
  const stop = useCallback(() => {
    release();
    if (!mountedRef.current) return;
    setPhase("idle");
    setStatus("Camera is off");
    setScores(null);
    setBox(null);
    setFaceCount(0);
    setLatency(null);
  }, [release]);
  useEffect(() => {
    mountedRef.current = true;
    const onHidden = () => {
      if (document.hidden) stop();
    };
    document.addEventListener("visibilitychange", onHidden);
    return () => {
      mountedRef.current = false;
      document.removeEventListener("visibilitychange", onHidden);
      release();
    };
  }, [release, stop]);

  const start = async () => {
    release();
    const token = tokenRef.current;
    const current = () => mountedRef.current && tokenRef.current === token;
    setError("");
    setPhase("loading");
    setStatus("Loading browser models…");
    setScores(null);
    setBox(null);
    setHistory([]);
    historyRef.current = [];
    setElapsed(0);
    setLatency(null);
    setFaceCount(0);
    try {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia)
        throw Error(
          "Camera access needs HTTPS or localhost and a browser with camera support.",
        );
      const model = await loadCameraModel();
      if (!current()) return;
      setStatus("Waiting for camera permission…");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });
      if (!current()) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = stream;
      stream.getVideoTracks().forEach((track) => {
        track.onended = () => {
          if (current()) stop();
        };
      });
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      if (!current()) return;
      startedRef.current = performance.now();
      setPhase("running");
      setStatus("Position one face in the frame");
      const options = new model.TinyFaceDetectorOptions({
        inputSize: 320,
        scoreThreshold: 0.5,
      });
      const frame = document.createElement("canvas");
      const context = frame.getContext("2d", { willReadFrequently: true });
      if (!context)
        throw Error(
          "This browser cannot prepare camera frames for evaluation.",
        );
      const tick = async () => {
        if (!current() || !video.videoWidth || video.paused) return;
        const began = performance.now();
        try {
          // Freeze one frame so face counting and expression inference use the
          // same image, even if someone moves while the model is running.
          frame.width = Math.min(640, video.videoWidth);
          frame.height = Math.round(
            (frame.width * video.videoHeight) / video.videoWidth,
          );
          context.drawImage(video, 0, 0, frame.width, frame.height);
          // Detection first: do not classify when multiple faces are detected.
          const faces = await model.detectAllFaces(frame, options);
          if (!current()) return;
          const seconds = (began - startedRef.current) / 1000;
          setElapsed((performance.now() - startedRef.current) / 1000);
          setSize({ width: frame.width, height: frame.height });
          setFaceCount(faces.length);
          if (faces.length !== 1) {
            setScores(null);
            setBox(null);
            setStatus(
              faces.length
                ? "Multiple faces detected — use one person at a time"
                : "No face detected — face the camera in good light",
            );
          } else {
            const result = await model
              .detectSingleFace(frame, options)
              .withFaceExpressions();
            if (!current()) return;
            if (result) {
              const values = Object.fromEntries(
                ORDER.map((key) => [key, Number(result.expressions[key] || 0)]),
              );
              const sorted = ORDER.slice().sort(
                (a, b) => values[b] - values[a],
              );
              const entry = {
                seconds: Number(seconds.toFixed(2)),
                expression: sorted[0],
                scores: values,
              };
              historyRef.current = [...historyRef.current, entry].slice(-300);
              setHistory(historyRef.current);
              setScores(values);
              setBox(result.detection.box);
              setStatus("One face · facial-expression estimates");
            } else {
              setScores(null);
              setBox(null);
              setStatus("Face lost — hold still and face the camera");
            }
          }
          setLatency(Math.round(performance.now() - began));
        } catch (e) {
          if (!current()) return;
          release();
          setPhase("idle");
          setScores(null);
          setBox(null);
          setFaceCount(0);
          setStatus("Evaluation stopped");
          setError(
            "Expression inference failed. Restart the camera to try again.",
          );
          return;
        }
        if (current())
          timerRef.current = setTimeout(
            tick,
            Math.max(100, 650 - (performance.now() - began)),
          );
      };
      timerRef.current = setTimeout(tick, 100);
    } catch (e) {
      if (!current()) return;
      release();
      setPhase("idle");
      setStatus("Camera is off");
      setError(messageFor(e));
    }
  };
  const exportResults = () => {
    const data = {
      kind: "local_facial_expression_estimates",
      model: "@vladmandic/face-api@1.7.15 faceExpressionNet",
      scope:
        "Facial appearance only; not internal emotions, mental state, or diagnosis.",
      retention: "Latest 300 observations from this browser session only.",
      samples: historyRef.current,
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "telesense-expression-estimates.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const ranked = scores
    ? ORDER.slice().sort((a, b) => scores[b] - scores[a])
    : ORDER;
  const top = scores ? ranked[0] : null;
  const busy = phase === "loading";
  return (
    <section className="camera-workspace">
      <div className="page-heading">
        <div>
          <h1>Camera context</h1>
          <p>
            A single-camera workspace for exploring visible facial-expression
            patterns. All evaluation happens on this device.
          </p>
        </div>
        <div className="camera-actions">
          <button
            className="button secondary"
            onClick={exportResults}
            disabled={!history.length}
          >
            <Icon name="download" size={15} />
            Export estimates
          </button>
          {phase === "running" || busy ? (
            <button className="button secondary" onClick={stop}>
              <Icon name="close" size={15} />
              {busy ? "Cancel" : "Stop camera"}
            </button>
          ) : (
            <button className="button primary" onClick={start}>
              <Icon name="video" size={16} />
              Start camera
            </button>
          )}
        </div>
      </div>
      <div className="camera-boundary">
        <Icon name="info" size={16} />
        <p>
          These are estimates of facial appearance, not someone’s internal
          emotions. The camera cannot establish mood, stress, intent, mental
          health, or a diagnosis.
        </p>
      </div>
      {error && (
        <div className="error-banner" role="alert">
          <Icon name="info" />
          <span>{error}</span>
        </div>
      )}
      <div className="camera-grid">
        <section className="camera-preview-panel">
          <div className="camera-panel-heading">
            <span>
              <i className={phase === "running" ? "camera-live-dot" : ""} />
              {phase === "running"
                ? "Camera on"
                : busy
                  ? "Preparing camera"
                  : "Camera off"}
            </span>
            <span>No microphone · no recording</span>
          </div>
          <div className="camera-stage">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className={
                phase === "running" ? "camera-video active" : "camera-video"
              }
              aria-label="Your camera preview"
            />
            {phase === "running" && box && (
              <svg
                className="camera-overlay"
                viewBox={`0 0 ${size.width} ${size.height}`}
                preserveAspectRatio="xMidYMid meet"
                aria-hidden="true"
              >
                <rect
                  x={box.x}
                  y={box.y}
                  width={box.width}
                  height={box.height}
                  rx="9"
                  fill="none"
                  stroke="#a6d3ba"
                  strokeWidth="2"
                />
              </svg>
            )}
            {phase !== "running" && (
              <div className="camera-placeholder">
                <div className="camera-lens">
                  <Icon name="video" size={36} />
                </div>
                <h2>
                  {busy
                    ? "Preparing local evaluation"
                    : "Your camera workspace"}
                </h2>
                <p>
                  {busy
                    ? status
                    : "Start the camera to see your preview and expression estimates."}
                </p>
                {!busy && (
                  <span>
                    Camera permission is requested only after you start.
                  </span>
                )}
              </div>
            )}
          </div>
          <div className="camera-status" role="status">
            <span>{status}</span>
            <span>
              {phase === "running" ? time(elapsed) : "—"}
              {latency !== null ? ` · ${latency} ms / evaluation` : ""}
            </span>
          </div>
        </section>
        <section className="camera-results-panel">
          <header>
            <span className="section-kicker">FACIAL APPEARANCE</span>
            <h2>Expression estimates</h2>
            <p>
              Model scores for the current frame. Scores are not calibrated
              probabilities of a person’s emotions.
            </p>
          </header>
          <div className="camera-top-result">
            <span>
              {top ? "Highest-scoring class" : "Waiting for one face"}
            </span>
            <strong>{top ? LABELS[top] : "No estimate"}</strong>
            <span>
              {top
                ? `${Math.round(scores[top] * 100)}% model score`
                : phase === "running" && faceCount > 1
                  ? "Multiple people — evaluation withheld"
                  : "Start the camera and face it directly"}
            </span>
          </div>
          <div className="expression-bars">
            {ranked.map((key) => (
              <div className="expression-row" key={key}>
                <span>{LABELS[key]}</span>
                <div>
                  <i
                    style={{
                      width: `${scores ? scores[key] * 100 : 0}%`,
                      background: COLORS[key],
                    }}
                  />
                </div>
                <span>
                  {scores ? `${Math.round(scores[key] * 100)}%` : "—"}
                </span>
              </div>
            ))}
          </div>
          <footer>
            <Icon name="shield" size={14} />
            <span>
              Face detection + expression classification only. No identity, age,
              or gender inference.
            </span>
          </footer>
        </section>
      </div>
      <section className="camera-history">
        <div className="panel-heading">
          <h3>Recent expression pattern</h3>
          <span>
            Latest {history.length} / 300 observations · local session only
          </span>
        </div>
        {history.length ? (
          <>
            <div
              className="camera-history-track"
              aria-label="Recent predicted facial-expression classes"
            >
              {history.map((entry, i) => (
                <span
                  key={`${entry.seconds}-${i}`}
                  style={{ background: COLORS[entry.expression] }}
                  title={`${time(entry.seconds)} · ${LABELS[entry.expression]} · ${Math.round(entry.scores[entry.expression] * 100)}%`}
                />
              ))}
            </div>
            <div className="camera-history-axis">
              <span>{time(history[0].seconds)}</span>
              <span>{time(history[history.length - 1].seconds)}</span>
            </div>
          </>
        ) : (
          <div className="camera-history-empty">
            Expression observations will appear here after a single face is
            detected.
          </div>
        )}
        <div className="camera-legend">
          {ORDER.map((key) => (
            <span key={key}>
              <i style={{ background: COLORS[key] }} />
              {LABELS[key]}
            </span>
          ))}
        </div>
      </section>
      <div className="camera-notes">
        <section>
          <h3>On-device by design</h3>
          <p>
            The browser downloads the face-detection and expression weights from
            this website. Frames are processed locally and are not uploaded to
            Fly. Only video is requested; the microphone stays off. Leaving this
            page, hiding the tab, or pressing Stop releases the camera.
          </p>
        </section>
        <section>
          <h3>Read facial patterns with care</h3>
          <p>
            Lighting, pose, glasses, occlusion, skin appearance, and
            training-data differences can affect results. A facial expression
            does not establish what someone feels. No detected face or a
            neutral-looking class is not a health assessment.
          </p>
          <a
            href="https://github.com/vladmandic/face-api"
            target="_blank"
            rel="noreferrer"
          >
            Model implementation
            <Icon name="arrow" size={13} />
          </a>
          <a
            href="/models/face-expression/LICENSE.txt"
            target="_blank"
            rel="noreferrer"
          >
            MIT license
          </a>
        </section>
      </div>
    </section>
  );
}
