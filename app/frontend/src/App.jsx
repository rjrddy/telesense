import { useCallback, useEffect, useRef, useState } from "react";
import Icon from "./components/Icon";
import CameraWorkspace from "./components/CameraWorkspace";
import SampleAttribution from "./components/SampleAttribution";
import ResearchGuide from "./components/ResearchGuide";
import ModelFlow from "./components/ModelFlow";
import SessionPlayer, { time } from "./components/SessionPlayer";
import {
  uploadVideo,
  startAnalysis,
  getStatus,
  getResults,
  authUrl,
} from "./services/api";
import "./styles/App.css";

const readDemoKey = () => {
  try {
    return localStorage.getItem("telesense_demo_key") || "";
  } catch {
    return "";
  }
};
const pretty = (value) => (value || "").replaceAll("_", " ");
const severityLabel = {
  high: "Strong signal",
  medium: "Moderate signal",
  low: "Subtle signal",
};
const hasBackend =
  Boolean(import.meta.env.VITE_API_URL) ||
  ["localhost", "127.0.0.1"].includes(window.location.hostname);

export default function App() {
  const [page, setPage] = useState("library");
  const [guideSection, setGuideSection] = useState("models");
  const openGuide = (section) => {
    setGuideSection(section);
    setPage("about");
  };
  const [samples, setSamples] = useState([]);
  const [selectedSampleId, setSelectedSampleId] = useState(null);
  const [session, setSession] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState(null);
  const [job, setJob] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [seek, setSeek] = useState(null);
  const [tab, setTab] = useState("transcript");
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [key, setKey] = useState(readDemoKey);
  const [keySaved, setKeySaved] = useState(false);
  const inputRef = useRef(null);
  const requestRef = useRef(0);
  const blobRef = useRef(null);
  const goTo = useCallback((t) => {
    setCurrentTime(t);
    setSeek({ time: t, id: Date.now() });
  }, []);
  const onTime = useCallback((t) => setCurrentTime(t), []);

  useEffect(() => {
    fetch("/demo/manifest.json")
      .then((r) => {
        if (!r.ok) throw Error();
        return r.json();
      })
      .then(setSamples)
      .catch(() =>
        setError(
          "Sample sessions could not be loaded. Please refresh to try again.",
        ),
      );
    return () => {
      if (blobRef.current) URL.revokeObjectURL(blobRef.current);
    };
  }, []);

  useEffect(() => {
    if (!job) return;
    let canceled = false;
    let timer;
    const check = async () => {
      try {
        const next = await getStatus(job);
        if (canceled) return;
        setStatus(next);
        if (next.status === "completed") {
          const data = await getResults(job);
          if (!canceled) {
            setResults(data);
            setLoading(false);
            setJob(null);
          }
        } else if (next.status === "failed") {
          setError(
            next.message || "Analysis failed. Please try another recording.",
          );
          setLoading(false);
          setJob(null);
        } else timer = setTimeout(check, 2000);
      } catch (e) {
        if (!canceled) {
          setError(
            e.response?.status === 401
              ? "Your demo key was rejected. Update it in Upload a recording."
              : "The analysis service is unreachable. You can still explore the sample library.",
          );
          setLoading(false);
          setJob(null);
        }
      }
    };
    check();
    return () => {
      canceled = true;
      clearTimeout(timer);
    };
  }, [job]);

  const openSample = async (sample) => {
    const request = ++requestRef.current;
    setSession(sample);
    setPage("review");
    setResults(null);
    setJob(null);
    setStatus(null);
    setCurrentTime(0);
    setSeek(null);
    setLoading(true);
    setError("");
    setSearch("");
    setTab("transcript");
    setFilter("all");
    try {
      const response = await fetch(sample.results);
      if (!response.ok) throw Error();
      const data = await response.json();
      if (request === requestRef.current) {
        setResults(data);
        setLoading(false);
      }
    } catch {
      if (request === requestRef.current) {
        setError(
          "This sample could not be opened. Return to the library and try again.",
        );
        setLoading(false);
      }
    }
  };

  const upload = async (file) => {
    if (!file) return;
    if (file.size > 99 * 1024 * 1024) {
      setError("Please choose a recording under 99 MB.");
      return;
    }
    if (!/\.(mp4|mov|avi|mkv|wav|mp3|m4a|flac)$/i.test(file.name)) {
      setError("Choose MP4, MOV, AVI, MKV, WAV, MP3, M4A, or FLAC.");
      return;
    }
    const request = ++requestRef.current;
    if (blobRef.current) URL.revokeObjectURL(blobRef.current);
    const media = URL.createObjectURL(file);
    blobRef.current = media;
    setSession({
      title: file.name,
      type: /\.(wav|mp3|m4a|flac)$/i.test(file.name) ? "audio" : "video",
      media,
      live: true,
    });
    setResults(null);
    setCurrentTime(0);
    setSeek(null);
    setError("");
    setLoading(true);
    setJob(null);
    setPage("review");
    setTab("transcript");
    setStatus({ progress: 0, message: "Uploading your recording…" });
    try {
      const uploaded = await uploadVideo(file);
      const next = await startAnalysis(uploaded.video_id);
      if (request === requestRef.current) {
        setStatus(next);
        setJob(next.job_id);
      }
    } catch (e) {
      if (request !== requestRef.current) return;
      setError(
        e.response?.status === 401
          ? "Enter a valid shared demo key in Upload a recording, then try again."
          : e.response?.data?.detail ||
              "Upload failed. Check that the analysis backend is available.",
      );
      setLoading(false);
    }
  };

  const exportResults = () => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(results, null, 2)], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `telesense-${session.id || "session"}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const signals = (results?.combined_flags || [])
    .slice()
    .sort((a, b) => a.start_time - b.start_time);
  const visibleSignals =
    filter === "all"
      ? signals
      : (results?.[`${filter}_flags`] || [])
          .slice()
          .sort((a, b) => a.start_time - b.start_time);
  const transcript = (results?.transcript || []).filter((s) =>
    s.text.toLowerCase().includes(search.toLowerCase()),
  );
  const duration = results?.duration || session?.duration || 0;
  const audio = session?.live
    ? results?.audio_url
      ? authUrl(results.audio_url)
      : session.type === "audio"
        ? session.media
        : null
    : session?.audio;
  const isReview = page === "review" && session;
  const selectedSample =
    samples.find((sample) => sample.id === selectedSampleId) || samples[0];

  return (
    <div className="app-shell">
      <header className="workspace-header">
        <a
          href="#"
          className="brand"
          onClick={(e) => {
            e.preventDefault();
            setPage("library");
          }}
        >
          <span className="brand-mark">
            <Icon name="wave" size={25} />
          </span>
          Telesense
        </a>

        <nav aria-label="Main navigation">
          <button
            className={page === "library" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("library")}
          >
            <Icon name="grid" />
            Sample library
            <span className="nav-count">{samples.length || "—"}</span>
          </button>
          <button
            className={page === "camera" ? "nav-item active" : "nav-item"}
            onClick={() => {
              setError("");
              setPage("camera");
            }}
          >
            <Icon name="video" />
            Camera context
          </button>
          {session && (
            <button
              className={isReview ? "nav-item active" : "nav-item"}
              onClick={() => setPage("review")}
            >
              <Icon name="wave" />
              Session review
            </button>
          )}
          <button
            className={page === "upload" ? "nav-item active" : "nav-item"}
            onClick={() => {
              setPage("upload");
              setError("");
            }}
          >
            <Icon name="upload" />
            Upload a recording
          </button>
          <button
            className={page === "about" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("about")}
          >
            <Icon name="info" />
            Models & context
          </button>
        </nav>
        <span className="workspace-status">
          <span />
          Research workspace
        </span>
      </header>

      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <Icon name="chevron" size={13} />
            <strong>
              {isReview
                ? "Session review"
                : {
                    library: "Sample library",
                    camera: "Camera context",
                    upload: "Upload a recording",
                    about: "Models & context",
                  }[page]}
            </strong>
          </div>
          <span className="research-badge">
            <span className="status-dot" />
            Research only
          </span>
        </header>
        <main>
          {error && (
            <div className="error-banner" role="alert">
              <Icon name="info" />
              <span>{error}</span>
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <Icon name="close" size={16} />
              </button>
            </div>
          )}
          {page === "camera" && <CameraWorkspace />}
          {page === "library" && (
            <section className="session-library">
              <div className="page-heading">
                <div>
                  <h1>Sample sessions</h1>
                  <p>
                    Recorded excerpts with saved analysis. Select a session to
                    review its transcript and signals.
                  </p>
                </div>
                <button
                  className="button primary"
                  onClick={() => {
                    setPage("upload");
                    setError("");
                  }}
                >
                  <Icon name="upload" size={16} />
                  Upload recording
                </button>
              </div>
              <div className="library-workbench">
                <section
                  className="recording-rail"
                  aria-label="Choose a sample"
                >
                  <div className="rail-heading">
                    <span>Recordings</span>
                    <span>{samples.length}</span>
                  </div>
                  <div className="recording-options">
                    {samples.map((sample, i) => (
                      <button
                        key={sample.id}
                        className={`recording-option ${selectedSample?.id === sample.id ? "selected" : ""}`}
                        aria-pressed={selectedSample?.id === sample.id}
                        onClick={() => setSelectedSampleId(sample.id)}
                      >
                        <div className="option-preview">
                          {sample.poster ? (
                            <img src={sample.poster} alt="" />
                          ) : (
                            <Icon name="wave" size={26} />
                          )}
                          <span>{time(sample.duration)}</span>
                        </div>
                        <div className="option-content">
                          <span className="option-index">
                            0{i + 1} /{" "}
                            {sample.type === "video"
                              ? "VIDEO + AUDIO"
                              : "AUDIO"}
                          </span>
                          <strong>{sample.title}</strong>
                          <p>{sample.media.split("/").pop()}</p>
                        </div>
                        <Icon name="chevron" size={16} />
                      </button>
                    ))}
                  </div>
                  {!samples.length && (
                    <div className="loading-panel">Loading recordings…</div>
                  )}
                  <div className="rail-note">
                    <Icon name="check" size={14} />
                    <span>
                      Saved analysis
                      <br />
                      No key required
                    </span>
                  </div>
                </section>
                <section
                  className="output-inspector"
                  aria-label="Selected sample outputs"
                >
                  <div className="inspector-heading">
                    <div>
                      <span className="section-kicker">OUTPUT INSPECTOR</span>
                      <h2>{selectedSample?.title || "Select a recording"}</h2>
                    </div>
                    <button
                      className="button primary"
                      disabled={!selectedSample}
                      onClick={() => openSample(selectedSample)}
                    >
                      Open session
                      <Icon name="arrow" size={16} />
                    </button>
                  </div>
                  {selectedSample && (
                    <ModelFlow
                      key={selectedSample.id}
                      sample={selectedSample}
                      onModels={() => openGuide("models")}
                    />
                  )}
                  <SampleAttribution sample={selectedSample} />
                  <div className="inspector-bottom">
                    <span>
                      <Icon name="clock" size={14} />
                      {time(selectedSample?.duration)} recording
                    </span>
                    <span>Pretrained pipeline · saved outputs</span>
                  </div>
                </section>
              </div>
              <div className="library-details">
                <section className="library-context">
                  <span className="section-kicker">Interpretation</span>
                  <h2>What does a concern label actually mean?</h2>
                  <p>
                    The model predicts emotions. Application rules map some
                    predictions to labels such as Depression or Anxiety. Those
                    labels describe the rule that fired, not a clinical
                    diagnosis.
                  </p>
                  <button
                    className="text-button"
                    onClick={() => openGuide("clinical")}
                  >
                    Read the diagnostic context
                    <Icon name="arrow" size={15} />
                  </button>
                </section>
                <section className="library-context data-context">
                  <span className="section-kicker">
                    Expanding the collection
                  </span>
                  <h2>New sources, different kinds of evidence.</h2>
                  <p>
                    Compare the sources behind the sample library: read speech,
                    acted emotion, multimodal performances, and restricted
                    research interviews. Each has its own reuse terms.
                  </p>
                  <button
                    className="text-button"
                    onClick={() => openGuide("data")}
                  >
                    Browse sample data sources
                    <Icon name="arrow" size={15} />
                  </button>
                </section>
              </div>
              <p className="library-disclaimer">
                <Icon name="info" size={16} />
                Research prototype. Do not use these outputs for diagnosis, risk
                assessment, or clinical decisions.
              </p>
            </section>
          )}

          {isReview && (
            <>
              <div className="review-title">
                <div>
                  <button
                    className="back-link"
                    onClick={() => setPage("library")}
                  >
                    <Icon name="back" size={15} />
                    Sample library
                  </button>
                  <h1>{session.title}</h1>
                  <p>
                    {session.live ? "Uploaded recording" : "Sample session"}
                    <span>·</span>
                    {time(duration)}
                    <span>·</span>
                    {session.type === "video"
                      ? "Video + audio + text"
                      : "Audio + text"}
                  </p>
                </div>
                <button
                  className="button secondary"
                  disabled={!results}
                  onClick={exportResults}
                >
                  <Icon name="download" size={16} />
                  Export analysis
                </button>
              </div>
              <div className="session-notice">
                <Icon name="info" size={16} />
                <span>
                  {session.live
                    ? "Live analysis · pretrained models on the backend"
                    : "Saved analysis · pretrained speech and language models"}
                  <strong>Research only</strong>
                </span>
              </div>
              {!session.live && <SampleAttribution sample={session} />}
              {loading && (
                <div className="processing-panel" role="status">
                  <span>{status?.message || "Opening saved analysis…"}</span>
                  <span>
                    {status ? `${Math.round(status.progress * 100)}%` : ""}
                  </span>
                  <div className="progress-track">
                    <i
                      style={{
                        width: `${status ? Math.max(3, status.progress * 100) : 40}%`,
                      }}
                    />
                  </div>
                </div>
              )}
              {results && (
                <details className="review-pipeline">
                  <summary>
                    <Icon name="wave" size={16} />
                    <span>Model output diagram</span>
                    <span className="pipeline-summary-count">
                      {signals.length} combined signals
                    </span>
                    <Icon name="chevron" size={14} />
                  </summary>
                  <ModelFlow
                    results={results}
                    title={session.title}
                    onModels={() => openGuide("models")}
                  />
                </details>
              )}
              <div className="review-grid">
                <div className="review-main">
                  <SessionPlayer
                    key={session.media}
                    url={session.media}
                    audio={audio}
                    type={session.type}
                    poster={session.poster}
                    title={session.title}
                    duration={duration}
                    currentTime={currentTime}
                    onTime={onTime}
                    seek={seek}
                  />
                  <section className="panel timeline-panel">
                    <div className="panel-heading">
                      <h3>Signal timeline</h3>
                      <span>Select a signal to seek</span>
                    </div>
                    <div className="timeline-rows">
                      {["audio", "text", "visual"].map((modality, i) => (
                        <div className="timeline-row" key={modality}>
                          <span>
                            <Icon
                              name={["wave", "text", "video"][i]}
                              size={14}
                            />
                            {["Vocal", "Language", "Visual"][i]}
                          </span>
                          <div className="timeline-lane">
                            {(results?.[`${modality}_flags`] || []).map(
                              (f, j) => (
                                <button
                                  key={j}
                                  title={`${pretty(f.concern_name)} · ${time(f.start_time)}`}
                                  aria-label={`Seek to ${pretty(f.concern_name)} at ${time(f.start_time)}`}
                                  className={`timeline-block ${f.severity}`}
                                  style={{
                                    left: `${(f.start_time / duration) * 100}%`,
                                    width: `${Math.max(0.8, ((f.end_time - f.start_time) / duration) * 100)}%`,
                                  }}
                                  onClick={() => goTo(f.start_time)}
                                />
                              ),
                            )}
                            <i
                              className="timeline-cursor"
                              style={{
                                left: `${duration ? (currentTime / duration) * 100 : 0}%`,
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="timeline-key">
                      <span>
                        <i className="low" />
                        Subtle
                      </span>
                      <span>
                        <i className="medium" />
                        Moderate
                      </span>
                      <span>
                        <i className="high" />
                        Strong
                      </span>
                      <small>
                        Visual emotion model not enabled for saved samples.
                      </small>
                    </div>
                  </section>
                </div>
                <div className="review-side">
                  <section className="panel insights-panel">
                    <div
                      className="insight-tabs"
                      role="tablist"
                      aria-label="Session insights"
                    >
                      {["transcript", "signals", "overview"].map((t) => (
                        <button
                          role="tab"
                          aria-selected={tab === t}
                          className={tab === t ? "active" : ""}
                          key={t}
                          onClick={() => setTab(t)}
                        >
                          {t}
                          {t === "signals" && <span>{signals.length}</span>}
                        </button>
                      ))}
                    </div>
                    {tab === "transcript" && (
                      <>
                        <div className="search-field">
                          <Icon name="search" size={16} />
                          <input
                            aria-label="Search transcript"
                            placeholder="Search transcript"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                          />
                        </div>
                        <div className="transcript-list">
                          {transcript.length ? (
                            transcript.map((s, i) => (
                              <button
                                className={`transcript-line ${currentTime >= s.start_time && currentTime < s.end_time ? "current" : ""}`}
                                key={i}
                                onClick={() => goTo(s.start_time)}
                              >
                                <span className="transcript-time">
                                  {time(s.start_time)}
                                  <Icon name="play" size={10} />
                                </span>
                                <div>
                                  <span className="speaker-label">
                                    TRANSCRIPT
                                  </span>
                                  <p>{s.text}</p>
                                </div>
                              </button>
                            ))
                          ) : (
                            <div className="empty-state">
                              <Icon name="text" size={28} />
                              <h4>
                                {loading
                                  ? "Transcript is on its way"
                                  : search
                                    ? "No matching words"
                                    : "No speech transcribed"}
                              </h4>
                              <p>
                                {search
                                  ? "Try a different word or clear your search."
                                  : "Speech recognized in this recording will appear here."}
                              </p>
                            </div>
                          )}
                        </div>
                        <div className="panel-footnote">
                          <Icon name="clock" size={13} />
                          Click a passage to jump to that moment.
                        </div>
                      </>
                    )}
                    {tab === "signals" && (
                      <>
                        <div className="signal-filters">
                          {["all", "audio", "text", "visual"].map((f) => (
                            <button
                              key={f}
                              className={filter === f ? "selected" : ""}
                              onClick={() => setFilter(f)}
                            >
                              {f === "all" ? "All signals" : f}
                            </button>
                          ))}
                        </div>
                        <div className="signal-context">
                          <p>
                            These concern labels come from emotion mappings and
                            keyword rules.
                          </p>
                          <button onClick={() => openGuide("clinical")}>
                            What they mean for diagnosis
                            <Icon name="arrow" size={12} />
                          </button>
                        </div>
                        <div className="signals-list">
                          {visibleSignals.length ? (
                            visibleSignals.map((s, i) => (
                              <button
                                className="signal-item"
                                key={i}
                                onClick={() => goTo(s.start_time)}
                              >
                                <div>
                                  <span
                                    className={`signal-dot ${s.severity}`}
                                  />
                                  <strong>{pretty(s.concern_name)}</strong>
                                  <span>{time(s.start_time)}</span>
                                </div>
                                <p>
                                  {severityLabel[s.severity]}
                                  <span>·</span>
                                  {s.modality}
                                  <span>·</span>
                                  {Math.round(s.confidence * 100)}% model score
                                </p>
                              </button>
                            ))
                          ) : (
                            <div className="empty-state">
                              <Icon name="check" size={28} />
                              <h4>No signals in this view</h4>
                              <p>
                                Absence of flags is not evidence of absence of a
                                concern.
                              </p>
                            </div>
                          )}
                        </div>
                      </>
                    )}
                    {tab === "overview" && (
                      <div className="overview-content">
                        <span className="eyebrow muted">Session overview</span>
                        <div className="overview-stats">
                          <div>
                            <strong>{signals.length}</strong>
                            <span>Combined signals</span>
                          </div>
                          <div>
                            <strong>{time(duration)}</strong>
                            <span>Recording length</span>
                          </div>
                          <div>
                            <strong>{results?.transcript?.length || 0}</strong>
                            <span>Transcript passages</span>
                          </div>
                          <div>
                            <strong>{results ? "2" : "—"}</strong>
                            <span>Model modalities*</span>
                          </div>
                        </div>
                        <h4>Reading the results</h4>
                        <p>
                          Start with the recording. Use the transcript and
                          signal timeline to revisit moments in context. Model
                          confidence is a prediction score, not a calibrated
                          probability.
                        </p>
                        <div className="method-note">
                          <Icon name="info" size={18} />
                          <p>
                            Saved results use speech and language models. Visual
                            emotion detection is unavailable. *Video playback
                            provides visual context.
                          </p>
                        </div>
                        <h4>Model stack</h4>
                        <p className="overview-guide-copy">
                          Whisper transcribes speech, Wav2Vec2 scores vocal
                          emotion, and RoBERTa scores emotion in language.
                          Concern labels are application rules, not diagnostic
                          predictions.
                        </p>
                        <div className="model-tags">
                          <span>Whisper</span>
                          <span>Wav2Vec2</span>
                          <span>RoBERTa</span>
                        </div>
                        <button
                          className="text-button guide-link"
                          onClick={() => openGuide("models")}
                        >
                          Model details and limitations{" "}
                          <Icon name="arrow" size={14} />
                        </button>
                        <button
                          className="text-button guide-link"
                          onClick={() => openGuide("clinical")}
                        >
                          How concern labels relate to diagnosis{" "}
                          <Icon name="arrow" size={14} />
                        </button>
                      </div>
                    )}
                  </section>
                </div>
              </div>
              <p className="review-disclaimer">
                <Icon name="shield" size={14} />
                Exploratory outputs, not clinical findings. Signals may be
                inaccurate and should always be reviewed in context.
              </p>
            </>
          )}

          {page === "upload" && (
            <section className="upload-page">
              <h1>Upload recording</h1>
              <p>
                Analyze a synthetic or public demo recording with the pretrained
                pipeline.
              </p>
              <div className="upload-card">
                <div className="upload-heading">
                  <span className="tag-icon green">
                    <Icon name="upload" size={24} />
                  </span>
                  <h2>Upload a recording</h2>
                </div>
                {!hasBackend && (
                  <div className="backend-notice">
                    <Icon name="info" />
                    <p>
                      <strong>Live uploads are not connected yet.</strong> The
                      public sample library is ready to explore. New recordings
                      need the analysis backend, which is being prepared
                      separately.
                    </p>
                  </div>
                )}
                <form
                  className="key-form"
                  onSubmit={(e) => {
                    e.preventDefault();
                    try {
                      localStorage.setItem("telesense_demo_key", key);
                      setKeySaved(true);
                    } catch {
                      setError(
                        "Browser storage is disabled. Allow local storage to save the demo key.",
                      );
                    }
                  }}
                >
                  <label htmlFor="demo-key">
                    Shared demo key<span>Required for new uploads</span>
                  </label>
                  <div>
                    <input
                      id="demo-key"
                      type="password"
                      value={key}
                      onChange={(e) => {
                        setKey(e.target.value);
                        setKeySaved(false);
                      }}
                      placeholder="Enter your access key"
                      autoComplete="off"
                    />
                    <button className="button secondary" type="submit">
                      {keySaved ? "Saved" : "Save key"}
                    </button>
                  </div>
                </form>
                <div
                  className={`upload-drop ${!hasBackend ? "disabled" : ""}`}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    if (hasBackend && !loading) upload(e.dataTransfer.files[0]);
                  }}
                >
                  <Icon name="upload" size={30} />
                  <h3>Drop your recording here</h3>
                  <p>or choose a file from your computer</p>
                  <button
                    className="button primary"
                    disabled={!hasBackend || loading}
                    onClick={() => inputRef.current.click()}
                  >
                    Choose recording
                    <Icon name="arrow" size={16} />
                  </button>
                  <input
                    ref={inputRef}
                    type="file"
                    accept=".mp4,.mov,.avi,.mkv,.wav,.mp3,.m4a,.flac"
                    hidden
                    onChange={(e) => {
                      upload(e.target.files[0]);
                      e.target.value = "";
                    }}
                  />
                  <span>
                    MP4, MOV, AVI, MKV, WAV, MP3, M4A, FLAC · Up to 99 MB
                  </span>
                </div>
                <p className="upload-privacy">
                  <Icon name="shield" size={16} />
                  Use synthetic or public recordings only. Do not upload patient
                  data. Uploaded recordings are retained on the demo server
                  until manually removed.
                </p>
              </div>
              <button
                className="text-button"
                onClick={() => setPage("library")}
              >
                <Icon name="back" size={16} />
                View sample sessions
              </button>
            </section>
          )}

          {page === "about" && (
            <ResearchGuide section={guideSection} onSection={setGuideSection} />
          )}
        </main>
        <footer className="app-footer">
          <span>
            Telesense <span>Session analysis</span>
          </span>
          <span>Audio · Language · Visual context</span>
        </footer>
      </div>
    </div>
  );
}
