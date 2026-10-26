import { useEffect, useId, useState } from "react";
import Icon from "./Icon";

export default function ModelFlow({ samples = [], results, title, onModels }) {
  const [preview, setPreview] = useState(null);
  const marker = useId().replaceAll(":", "");
  useEffect(() => {
    if (results || !samples[0]) return;
    const controller = new AbortController();
    fetch(samples[0].results, { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw Error();
        return r.json();
      })
      .then(setPreview)
      .catch(() => {});
    return () => controller.abort();
  }, [results, samples]);
  const data = results || preview;
  const transcript = data?.transcript?.[0]?.text;
  const audioFlag = data?.audio_flags?.[0];
  const textFlag = data?.text_flags?.[0];
  const score = (flag) =>
    flag ? `${Math.round(flag.confidence * 100)}%` : "—";
  const label = (flag) => flag?.concern_name?.replaceAll("_", " ") || "No flag";
  return (
    <section className="model-flow" aria-label="Model output pipeline">
      <header className="flow-heading">
        <div>
          <span className="section-kicker">MODEL OUTPUTS</span>
          <h2>From recording to review.</h2>
        </div>
        <span className="flow-source">
          {results
            ? title
            : samples[0]
              ? `Saved sample · ${samples[0].title}`
              : "Loading sample output…"}
        </span>
      </header>
      <div className="flow-canvas">
        <svg
          className="flow-connectors"
          viewBox="0 0 1000 320"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <defs>
            <marker
              id={marker}
              markerWidth="7"
              markerHeight="7"
              refX="6"
              refY="3.5"
              orient="auto"
            >
              <path
                d="M1 1L6 3.5 1 6"
                fill="none"
                stroke="#7fbeb1"
                strokeWidth="1"
              />
            </marker>
          </defs>
          {[
            "M160 160C195 160 195 77 230 77",
            "M160 160C195 160 195 247 230 247",
            "M450 77L515 77",
            "M735 77C780 77 780 145 810 145",
            "M735 247C780 247 780 195 810 195",
          ].map((d, i) => (
            <g key={d}>
              <path
                className="connector-track"
                d={d}
                markerEnd={`url(#${marker})`}
              />
              <path
                className="connector-pulse"
                d={d}
                style={{ animationDelay: `${i * -0.8}s` }}
              />
            </g>
          ))}
        </svg>
        <div className="flow-node flow-input">
          <Icon name="video" size={21} />
          <span className="flow-label">INPUT</span>
          <h3>Recording</h3>
          <p>
            Audio waveform
            <br />
            Video as context
          </p>
        </div>
        <button className="flow-node flow-whisper" onClick={onModels}>
          <span className="flow-node-top">
            <Icon name="file" size={17} />
            <span>01 · TRANSCRIBE</span>
          </span>
          <h3>Whisper base</h3>
          <p className="flow-quote">
            {transcript ? `“${transcript}”` : "Timestamped transcript"}
          </p>
          <span className="flow-meta">
            {data ? `${data.transcript.length} passages` : "Speech → text"}
          </span>
        </button>
        <button className="flow-node flow-roberta" onClick={onModels}>
          <span className="flow-node-top">
            <Icon name="text" size={17} />
            <span>02 · LANGUAGE</span>
          </span>
          <h3>RoBERTa</h3>
          <div className="flow-score">
            <strong>{score(textFlag)}</strong>
            <span>{data ? label(textFlag) : "Awaiting output"}</span>
          </div>
          <span className="flow-meta">First language flag · model score</span>
        </button>
        <button className="flow-node flow-wav2vec" onClick={onModels}>
          <span className="flow-node-top">
            <Icon name="wave" size={17} />
            <span>PARALLEL · VOCAL EXPRESSION</span>
          </span>
          <div className="flow-audio-output">
            <div>
              <h3>Wav2Vec2 · SUPERB</h3>
              <p>3-second windows / 0.5-second hop</p>
            </div>
            <div className="flow-score">
              <strong>{score(audioFlag)}</strong>
              <span>{data ? label(audioFlag) : "Awaiting output"}</span>
            </div>
          </div>
          <span className="flow-meta">First audio flag · model score</span>
        </button>
        <div className="flow-node flow-result">
          <span className="flow-label">APPLICATION RULES</span>
          <h3>Review signals</h3>
          <strong>{data ? data.combined_flags?.length || 0 : "—"}</strong>
          <p>
            Combined flags
            <br />
            Not diagnoses
          </p>
        </div>
      </div>
      <footer>
        <span className="flow-motion-key">
          <i />
          Animated connectors show data flow, not live processing.
        </span>
        <span>Scores are uncalibrated · select a model for details</span>
      </footer>
    </section>
  );
}
