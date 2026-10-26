import React from 'react'
import './ProtectiveFactorsPanel.css'

const ProtectiveFactorsPanel = ({ evaluations, onSeek }) => {
  if (!evaluations || evaluations.length === 0) {
    return (
      <div className="protective-factors-panel">
        <h4>Protective Factors</h4>
        <p className="no-data">No evaluation data available</p>
      </div>
    )
  }

  // Collect all protective factors from all evaluations
  const allFactors = []
  evaluations.forEach((evaluation, evalIdx) => {
    const protectiveFactors = evaluation.protective_factors || []
    if (protectiveFactors.length > 0) {
      protectiveFactors.forEach((factor, factorIdx) => {
        allFactors.push({
          ...factor,
          evaluationName: evaluation.name,
          evaluationBasis: evaluation.basis
        })
      })
    }
  })

  if (allFactors.length === 0) {
    return (
      <div className="protective-factors-panel">
        <h4>Protective Factors</h4>
        <p className="no-data">No protective factors identified</p>
      </div>
    )
  }

  // Sort by timestamp if available
  allFactors.sort((a, b) => {
    const timeA = a.time_s || 0
    const timeB = b.time_s || 0
    return timeA - timeB
  })

  // Format time for display
  const formatTime = (seconds) => {
    if (typeof seconds !== 'number') return 'N/A'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="protective-factors-panel">
      <h4>Protective Factors</h4>
      <p className="panel-description">
        Positive indicators and strengths identified during the assessment.
      </p>

      <div className="factors-summary">
        <div className="summary-stat">
          <span className="stat-value">{allFactors.length}</span>
          <span className="stat-label">Total Protective Factors</span>
        </div>
      </div>

      <div className="factors-list">
        {allFactors.map((factor, idx) => (
          <div 
            key={idx} 
            className="protective-factor-card"
            onClick={() => {
              if (onSeek && typeof factor.time_s === 'number') {
                onSeek(factor.time_s)
              }
            }}
          >
            <div className="factor-header">
              <div className="factor-time">
                {factor.timestamp || formatTime(factor.time_s)}
                {factor.time_s && onSeek && (
                  <button 
                    className="seek-button"
                    onClick={(e) => {
                      e.stopPropagation()
                      onSeek(factor.time_s)
                    }}
                    title="Jump to timestamp"
                  >
                    ▶
                  </button>
                )}
              </div>
              <div className="factor-evaluation">
                Evaluation
              </div>
            </div>

            {factor.evidence && (
              <div className="factor-evidence">
                <span className="evidence-label">Evidence:</span>
                <blockquote className="evidence-quote">
                  "{factor.evidence}"
                </blockquote>
              </div>
            )}

            {factor.rationale && (
              <div className="factor-rationale">
                <span className="rationale-label">Rationale:</span>
                <p className="rationale-text">{factor.rationale}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default ProtectiveFactorsPanel
