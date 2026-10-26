import Icon from "./Icon";

export default function SampleAttribution({ sample }) {
  const source = sample?.attribution;
  if (!source) return null;
  return (
    <section
      className="sample-attribution"
      aria-label="Recording source and reference"
    >
      <div className="attribution-heading">
        <span>
          <Icon name="file" size={14} />
          <a href={source.url} target="_blank" rel="noreferrer">
            {source.dataset}
          </a>
          <span className="source-license">{source.license}</span>
        </span>
        <span>{source.source_label}</span>
      </div>
      <details>
        <summary>
          Reference transcript & source details
          <Icon name="chevron" size={12} />
        </summary>
        <div className="source-details">
          <h4>Corpus reference · not model output</h4>
          <blockquote>{source.reference_transcript}</blockquote>
          <p>
            The source label describes a performance or recording type. It is
            not a diagnosis or an endorsement of the model’s predictions.
          </p>
          <dl>
            <dt>Clip identifier</dt>
            <dd>{source.clip_id}</dd>
            <dt>Attribution</dt>
            <dd>{source.authors}</dd>
            <dt>Modifications</dt>
            <dd>{source.changes}</dd>
          </dl>
          <div className="attribution-links">
            <a href={source.license_url} target="_blank" rel="noreferrer">
              License terms
            </a>
            {source.contents_license_url && (
              <a
                href={source.contents_license_url}
                target="_blank"
                rel="noreferrer"
              >
                Contents license
              </a>
            )}
            {source.license_files.map((path) => (
              <a href={path} key={path} download>
                {path.split("/").pop()}
              </a>
            ))}
            {source.dataset === "CREMA-D" && (
              <a href="/demo/crema-d-subset.json" download>
                Download licensed subset catalog
              </a>
            )}
          </div>
        </div>
      </details>
    </section>
  );
}
