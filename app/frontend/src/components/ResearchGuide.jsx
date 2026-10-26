import Icon from "./Icon";
import { models, datasets } from "../data/research";
const Source = ({ href, children }) => (
  <a href={href} target="_blank" rel="noreferrer">
    {children}
    <Icon name="arrow" size={13} />
  </a>
);

export default function ResearchGuide({ section, onSection }) {
  return (
    <section className="research-guide">
      <div className="guide-heading">
        <span className="section-kicker">Behind the analysis</span>
        <h1>Models, signals, and clinical context</h1>
        <p>
          A closer look at what this prototype computes, how its labels are
          created, and what a recording cannot tell us.
        </p>
      </div>
      <div className="guide-tabs" role="tablist" aria-label="Research guide">
        {[
          ["models", "Model guide"],
          ["clinical", "Diagnosis & context"],
          ["data", "Sample data sources"],
        ].map(([id, label]) => (
          <button
            key={id}
            role="tab"
            aria-selected={section === id}
            onClick={() => onSection(id)}
          >
            {label}
          </button>
        ))}
      </div>
      {section === "models" && (
        <div className="guide-body">
          <div className="guide-intro">
            <h2>Three pretrained models. Different jobs.</h2>
            <p>
              The public demo runs the pretrained fallback on CPU. It does not
              use a trained Telesense checkpoint or a diagnostic classifier.
            </p>
          </div>
          <div className="model-detail-list">
            {models.map((m, i) => (
              <article className={`model-detail tone-${m.tone}`} key={m.id}>
                <header>
                  <span className="model-icon">
                    <Icon name={m.icon} size={22} />
                  </span>
                  <div>
                    <span className="section-kicker">{m.role}</span>
                    <h3>{m.name}</h3>
                  </div>
                  <span className="model-index">0{i + 1}</span>
                </header>
                <code>{m.checkpoint}</code>
                <p>{m.description}</p>
                <div className="model-facts">
                  {m.facts.map((f) => (
                    <span key={f}>{f}</span>
                  ))}
                </div>
                <div className="model-explanation">
                  <div>
                    <h4>Training background</h4>
                    <p>{m.training}</p>
                  </div>
                  <div>
                    <h4>What to watch for</h4>
                    <p>{m.limitation}</p>
                  </div>
                </div>
                <Source href={m.source}>Read the model card</Source>
              </article>
            ))}
          </div>
          <section className="guide-note">
            <h3>What Telesense adds</h3>
            <p>
              Application rules turn emotion scores and keyword matches into
              timestamped concern flags. For example, language labeled sadness,
              grief, or disappointment can be mapped to “Depression.” This
              mapping is a project heuristic, not a trained or validated
              diagnosis. Overlapping windows can produce several flags for the
              same moment.
            </p>
            <p>
              Video in batch review is available as playback context; batch
              visual-expression inference remains disabled. The separate Camera
              context workspace estimates visible facial expressions locally in
              the browser. It does not infer internal emotions or clinical
              conditions. The repository also contains trainable multimodal and
              temporal models, but no trained checkpoint is used here.
            </p>
          </section>
          <section className="research-tracks">
            <h2>Research models in the repository</h2>
            <p>
              These are separate experimental tracks. They are not the engine
              serving the public demo.
            </p>
            <div>
              <article>
                <span className="section-kicker">
                  Supervised · checkpoint required
                </span>
                <h3>Temporal multimodal model</h3>
                <p>
                  Shared Wav2Vec2, ResNet-50 / DenseNet-121, and DistilBERT
                  encoders represent audio, frames, and text. A temporal
                  convolutional network and Transformer combine features over
                  time, with a per-frame head to localize labeled events.
                </p>
                <span className="inactive-model">
                  Not deployed · no trained checkpoint configured
                </span>
              </article>
              <article>
                <span className="section-kicker">
                  Unsupervised · experimental
                </span>
                <h3>Temporal VAE + contrastive learning</h3>
                <p>
                  A temporal variational autoencoder and contrastive objectives
                  explore structure in recordings without diagnostic labels.
                  Reconstruction or representation differences can suggest
                  unusual patterns, but do not identify a disorder.
                </p>
                <span className="inactive-model">
                  Not deployed · research track only
                </span>
              </article>
            </div>
          </section>
          <div className="guide-note score-note">
            <h3>Reading the numbers</h3>
            <p>
              A model score describes the model’s output for a label. It is not
              the probability that someone has a disorder. “Strong,” “moderate,”
              and “subtle” describe uncalibrated application thresholds; they do
              not measure clinical severity or risk. This demo does not report
              clinical sensitivity, specificity, or diagnostic accuracy.
            </p>
          </div>
        </div>
      )}
      {section === "clinical" && (
        <div className="guide-body">
          <div className="clinical-banner">
            <Icon name="shield" size={25} />
            <div>
              <h2>Emotion recognition is not diagnosis.</h2>
              <p>
                A sad-sounding voice, a fearful phrase, or a flagged keyword
                does not establish a disorder. These are starting points for
                inspecting model behavior, not findings about a person.
              </p>
            </div>
          </div>
          <div className="clinical-grid">
            <article className="clinical-card">
              <span className="section-kicker">Mood-related labels</span>
              <h3>Depression</h3>
              <p>
                In this app, the label can come from an emotion-to-concern rule
                or a keyword match. It does not indicate that clinical criteria
                have been met.
              </p>
              <h4>Clinical context</h4>
              <p>
                NIMH describes depression as involving persistent symptoms that
                affect daily life. Assessment considers duration, loss of
                interest or low mood, other symptoms, and possible medical or
                medication-related causes. A few seconds of emotion prediction
                cannot establish that pattern.
              </p>
              <Source href="https://www.nimh.nih.gov/health/publications/depression">
                NIMH: depression and diagnosis
              </Source>
            </article>
            <article className="clinical-card">
              <span className="section-kicker">Fear / nervousness labels</span>
              <h3>Anxiety</h3>
              <p>
                The app maps some language emotion scores, such as fear and
                nervousness, to an anxiety concern label. Normal worry,
                performance, and fictional dialogue can also produce those
                signals.
              </p>
              <h4>Clinical context</h4>
              <p>
                Anxiety disorders involve more than temporary worry. For
                generalized anxiety disorder, clinicians assess persistent,
                difficult-to-control worry, associated symptoms, and effects on
                daily life. A voice classifier cannot determine these from tone
                alone.
              </p>
              <Source href="https://www.nimh.nih.gov/health/publications/generalized-anxiety-disorder-gad">
                NIMH: generalized anxiety disorder
              </Source>
            </article>
          </div>
          <section className="label-reference">
            <h2>Other labels you may encounter</h2>
            <div>
              <strong>Agitation / negative emotion</strong>
              <p>
                Application mappings from emotions such as anger, annoyance, or
                disgust. These do not identify a specific disorder or explain
                why the speaker sounds that way.
              </p>
            </div>
            <div>
              <strong>Behavioral concern</strong>
              <p>
                A broad application label for a flagged window. It is not a
                named diagnosis and can occur in ordinary speech or acted
                material.
              </p>
            </div>
            <div>
              <strong>Self-harm or crisis-related language</strong>
              <p>
                Keyword rules may flag explicit phrases. They do not determine
                intent, current safety, speaker identity, or urgency, and cannot
                replace a professional risk assessment.
              </p>
            </div>
            <div>
              <strong>No flags</strong>
              <p>
                Means that the pipeline did not produce flags for the recording.
                It is not a negative clinical test and does not rule out
                distress or illness.
              </p>
            </div>
          </section>
          <div className="guide-note">
            <h3>What a clinical assessment needs</h3>
            <p>
              Symptoms over time, their impact on functioning, medical and
              psychiatric history, alternative explanations, and a qualified
              professional’s assessment. Telesense does not collect enough
              evidence to perform that process, and does not administer or score
              a validated screening instrument in this demo.
            </p>
          </div>
        </div>
      )}
      {section === "data" && (
        <div className="guide-body">
          <div className="guide-intro">
            <h2>Sources for a broader sample library</h2>
            <p>
              The library now includes three CREMA-D performances and two
              LibriSpeech utterances alongside the approved local excerpts.
              RAVDESS and DAIC-WOZ remain research candidates with different
              reuse constraints.
            </p>
          </div>
          <div className="dataset-grid">
            {datasets.map((d) => (
              <article className={`dataset-card tone-${d.tone}`} key={d.name}>
                <span className="dataset-status">{d.status}</span>
                <h3>{d.name}</h3>
                <span className="dataset-type">{d.type}</span>
                <p>{d.description}</p>
                <dl>
                  <dt>Purpose</dt>
                  <dd>{d.recommendation}</dd>
                  <dt>Published terms</dt>
                  <dd>{d.license}</dd>
                </dl>
                <p className="dataset-caveat">{d.caveat}</p>
                <Source href={d.url}>View the original source</Source>
              </article>
            ))}
          </div>
          <div className="guide-note">
            <h3>What is included now</h3>
            <p>
              Two LibriSpeech utterances include reference transcripts. Three
              CREMA-D videos contrast neutral, happy, and sad performances of
              the same sentence. Source labels and reference text are shown
              separately from predictions; neither is a clinical diagnosis.
            </p>
            <p>
              For every published clip, record its source URL, clip identifier,
              authors, license, and any edits or format conversions. Restricted
              interview recordings belong in approved private research workflows
              rather than this public portfolio.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
