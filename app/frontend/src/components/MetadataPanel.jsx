import React from 'react'
import './MetadataPanel.css'

const MetadataPanel = ({ metadata }) => {
  if (!metadata) {
    return (
      <div className="metadata-panel">
        <h4>Metadata</h4>
        <p className="no-data">No metadata available</p>
      </div>
    )
  }

  const filesProvided = metadata.files_provided || []
  const modalitiesUsed = metadata.modalities_used || []
  const notes = metadata.notes || []

  // Calculate data completeness
  const hasAudio = filesProvided.includes('audio') || modalitiesUsed.includes('audio')
  const hasVideo = filesProvided.includes('video') || modalitiesUsed.includes('video')
  const hasText = filesProvided.includes('transcript') || modalitiesUsed.includes('text')
  
  const completenessScore = (
    (hasAudio ? 1 : 0) + 
    (hasVideo ? 1 : 0) + 
    (hasText ? 1 : 0)
  ) / 3 * 100

  const getFileIcon = (fileType) => {
    const icons = {
      'audio': '🎵',
      'video': '🎬',
      'transcript': '📝',
      'text': '📄'
    }
    return icons[fileType.toLowerCase()] || '📄'
  }

  const getModalityBadgeColor = (modality) => {
    const colors = {
      'audio': '#3b82f6',
      'video': '#ef4444',
      'text': '#10b981'
    }
    return colors[modality.toLowerCase()] || '#6b7280'
  }

  return (
    <div className="metadata-panel">
      <h4>Analysis Metadata</h4>

      {/* Data Completeness */}
      <div className="completeness-section">
        <h5>Data Completeness</h5>
        <div className="completeness-score">
          <div className="score-circle" style={{ 
            background: `conic-gradient(#10b981 ${completenessScore * 3.6}deg, #e5e7eb 0deg)` 
          }}>
            <div className="score-inner">
              <span className="score-value">{Math.round(completenessScore)}%</span>
            </div>
          </div>
          <p className="score-label">Complete</p>
        </div>
        <div className="modality-indicators">
          {hasAudio && (
            <div className="modality-indicator available">
              <span className="indicator-icon">🎵</span>
              <span>Audio</span>
            </div>
          )}
          {hasVideo && (
            <div className="modality-indicator available">
              <span className="indicator-icon">🎬</span>
              <span>Video</span>
            </div>
          )}
          {hasText && (
            <div className="modality-indicator available">
              <span className="indicator-icon">📝</span>
              <span>Text</span>
            </div>
          )}
          {!hasAudio && (
            <div className="modality-indicator missing">
              <span className="indicator-icon">🎵</span>
              <span>Audio</span>
            </div>
          )}
          {!hasVideo && (
            <div className="modality-indicator missing">
              <span className="indicator-icon">🎬</span>
              <span>Video</span>
            </div>
          )}
          {!hasText && (
            <div className="modality-indicator missing">
              <span className="indicator-icon">📝</span>
              <span>Text</span>
            </div>
          )}
        </div>
      </div>

      {/* Files Provided */}
      {filesProvided.length > 0 && (
        <div className="metadata-section">
          <h5>Files Provided</h5>
          <div className="files-list">
            {filesProvided.map((file, idx) => (
              <div key={idx} className="file-item">
                <span className="file-icon">{getFileIcon(file)}</span>
                <span className="file-name">{file}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Modalities Used */}
      {modalitiesUsed.length > 0 && (
        <div className="metadata-section">
          <h5>Modalities Used</h5>
          <div className="modalities-list">
            {modalitiesUsed.map((modality, idx) => (
              <span 
                key={idx} 
                className="modality-badge"
                style={{ backgroundColor: getModalityBadgeColor(modality) }}
              >
                {modality}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Notes */}
      {notes.length > 0 && (
        <div className="metadata-section">
          <h5>Notes</h5>
          <div className="notes-list">
            {notes.map((note, idx) => (
              <div key={idx} className="note-item">
                <span className="note-icon">ℹ️</span>
                <p className="note-text">{note}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default MetadataPanel
