import { useEffect, useRef, useState } from "react";
import Icon from "./Icon";
export const time = (n = 0) =>
  `${Math.floor(n / 60)}:${String(Math.floor(n % 60)).padStart(2, "0")}`;

export default function SessionPlayer({
  url,
  audio,
  poster,
  type,
  duration,
  currentTime,
  onTime,
  seek,
  title,
}) {
  const ref = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState("");
  const [peaks, setPeaks] = useState([]);
  const [waveError, setWaveError] = useState(false);
  useEffect(() => {
    setPlaying(false);
    setError("");
    setPeaks([]);
    setWaveError(false);
    const controller = new AbortController();
    let context;
    if (audio)
      fetch(audio, { signal: controller.signal })
        .then((r) => {
          if (!r.ok) throw Error("Audio unavailable");
          return r.arrayBuffer();
        })
        .then(async (data) => {
          context = new AudioContext();
          const buffer = await context.decodeAudioData(data);
          const values = buffer.getChannelData(0);
          const step = Math.max(1, Math.floor(values.length / 160));
          const bars = Array.from({ length: 160 }, (_, i) => {
            let max = 0;
            for (
              let j = i * step;
              j < Math.min((i + 1) * step, values.length);
              j += 8
            )
              max = Math.max(max, Math.abs(values[j]));
            return max;
          });
          const max = Math.max(...bars, 0.001);
          if (!controller.signal.aborted) setPeaks(bars.map((v) => v / max));
        })
        .catch(() => {
          if (!controller.signal.aborted) setWaveError(true);
        })
        .finally(() => context?.close().catch(() => {}));
    return () => {
      controller.abort();
    };
  }, [url, audio]);
  useEffect(() => {
    if (ref.current && seek) {
      ref.current.currentTime = seek.time;
      onTime(seek.time);
    }
  }, [seek, onTime]);
  const toggle = async () => {
    if (!ref.current) return;
    try {
      if (ref.current.paused) await ref.current.play();
      else ref.current.pause();
    } catch {
      setError("Playback could not start. Please try again.");
    }
  };
  const props = {
    ref,
    src: url,
    onTimeUpdate: (e) => onTime(e.currentTarget.currentTime),
    onPlay: () => setPlaying(true),
    onPause: () => setPlaying(false),
    onEnded: () => setPlaying(false),
    onError: () => setError("This recording could not be loaded."),
    preload: "metadata",
  };
  return (
    <section className="session-player">
      <div className={`media-stage ${type === "audio" ? "audio-stage" : ""}`}>
        {type === "video" ? (
          <video {...props} poster={poster} playsInline onClick={toggle} />
        ) : (
          <>
            <audio {...props} />
            <div className={`audio-orbit ${playing ? "playing" : ""}`}>
              <Icon name="wave" size={50} />
            </div>
            <p>AUDIO SESSION</p>
            <h3>{title}</h3>
            <span>Use the controls below to play or seek.</span>
          </>
        )}
        <span className="media-label">
          <span className="status-dot" />{" "}
          {type === "video" ? "VIDEO + AUDIO" : "AUDIO RECORDING"}
        </span>
      </div>
      <div className="player-controls">
        <button
          className="play-button"
          onClick={toggle}
          aria-label={playing ? "Pause recording" : "Play recording"}
        >
          <Icon name={playing ? "pause" : "play"} size={18} />
        </button>
        <span className="player-time">
          {time(currentTime)} <em>/ {time(duration)}</em>
        </span>
        <input
          aria-label="Seek recording"
          type="range"
          min="0"
          max={duration || 1}
          step="0.05"
          value={Math.min(currentTime, duration || 0)}
          onChange={(e) => {
            ref.current.currentTime = Number(e.target.value);
            onTime(Number(e.target.value));
          }}
        />
        <span className="playback-speed">1×</span>
      </div>
      {error && (
        <p className="inline-error" role="alert">
          {error}
        </p>
      )}
      <div className="waveform-heading">
        <span>
          <Icon name="wave" size={15} /> Audio waveform
        </span>
        <span>Click to seek</span>
      </div>
      <button
        className="waveform"
        aria-label="Seek using audio waveform"
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const t = Math.max(
            0,
            Math.min(
              duration,
              e.detail === 0
                ? currentTime + 5
                : ((e.clientX - rect.left) / rect.width) * duration,
            ),
          );
          ref.current.currentTime = t;
          onTime(t);
        }}
      >
        {peaks.length ? (
          <svg
            viewBox="0 0 800 64"
            preserveAspectRatio="none"
            aria-hidden="true"
          >
            {peaks.map((v, i) => (
              <rect
                key={i}
                x={i * 5}
                y={32 - Math.max(2, v * 29)}
                width="2.5"
                height={Math.max(4, v * 58)}
                rx="1"
                fill={
                  i / peaks.length <= currentTime / duration
                    ? "#5278a8"
                    : "#c3cedd"
                }
              />
            ))}
          </svg>
        ) : (
          <span className="waveform-empty">
            {waveError
              ? "Waveform unavailable · playback is still available"
              : audio
                ? "Preparing waveform…"
                : "Waveform available after analysis"}
          </span>
        )}
        <i
          style={{ left: `${duration ? (currentTime / duration) * 100 : 0}%` }}
        />
      </button>
      <div className="waveform-axis">
        <span>0:00</span>
        <span>{time(duration / 2)}</span>
        <span>{time(duration)}</span>
      </div>
    </section>
  );
}
