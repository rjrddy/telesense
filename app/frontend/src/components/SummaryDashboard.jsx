import React, { useState } from 'react'
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import EmotionTimeline from './EmotionTimeline'
import MissingQuestionsPanel from './MissingQuestionsPanel'
import ProtectiveFactorsPanel from './ProtectiveFactorsPanel'
import './SummaryDashboard.css'

// Helper function to get display label for severity
const getSeverityLabel = (severity) => {
  if (severity === 'high') return 'High Risk'
  if (severity === 'medium') return 'High Concern'
  if (severity === 'low') return 'Concern'
  return severity // fallback
}

const SummaryDashboard = ({ results, onSeek }) => {
  const [activeTab, setActiveTab] = useState('overview')
  
  if (!results || !results.statistics) {
    return null
  }

  const stats = results.statistics
  const flagsBySeverity = stats.flags_by_severity || {}
  const flagsByType = stats.flags_by_type || {}
  const emotionSummary = results.emotion_summary
  const evaluations = results.evaluations || []

  const hasMissingQuestions = evaluations.some(evaluation => 
    evaluation.missing_critical_questions && 
    Array.isArray(evaluation.missing_critical_questions) && 
    evaluation.missing_critical_questions.length > 0
  )
  const hasProtectiveFactors = evaluations.some(evaluation => 
    evaluation.protective_factors && 
    Array.isArray(evaluation.protective_factors) && 
    evaluation.protective_factors.length > 0
  )

  // Prepare data for charts
  const severityData = [
    { name: getSeverityLabel('high'), value: flagsBySeverity.high || 0, color: '#ef4444' }, // High Risk: red
    { name: getSeverityLabel('medium'), value: flagsBySeverity.medium || 0, color: '#f59e0b' }, // High Concern: orange
    { name: getSeverityLabel('low'), value: flagsBySeverity.low || 0, color: '#fbbf24' }, // Concern: yellow
  ]

  const typeData = Object.entries(flagsByType).map(([name, value]) => ({
    name: name.length > 20 ? name.substring(0, 20) + '...' : name,
    value,
  })).sort((a, b) => b.value - a.value).slice(0, 10)

  const totalFlags = stats.total_flags || 0
  const duration = results.duration || 0
  const flagsPerMinute = duration > 0 ? (totalFlags / duration) * 60 : 0

  return (
    <div className="summary-dashboard">
      <h3>Analysis Summary</h3>
      {results.engine && (
        <p className="engine-label">
          Engine: {results.engine.name === 'telesense_model'
            ? `Telesense model (${results.engine.checkpoint})`
            : 'Pretrained emotion models'}
          {results.engine.note && <span className="engine-note"> · {results.engine.note}</span>}
        </p>
      )}

      {/* Tabs */}
      <div className="summary-tabs">
        <button 
          className={activeTab === 'overview' ? 'active' : ''}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        {emotionSummary && (
          <button 
            className={activeTab === 'emotions' ? 'active' : ''}
            onClick={() => setActiveTab('emotions')}
          >
            Emotion Timeline
          </button>
        )}
        {evaluations.length > 0 && (
          <button 
            className={activeTab === 'evaluations' ? 'active' : ''}
            onClick={() => setActiveTab('evaluations')}
          >
            Evaluation Details
          </button>
        )}
        {hasMissingQuestions && (
          <button 
            className={activeTab === 'missing-questions' ? 'active' : ''}
            onClick={() => setActiveTab('missing-questions')}
          >
            Missing Questions
          </button>
        )}
        {hasProtectiveFactors && (
          <button 
            className={activeTab === 'protective-factors' ? 'active' : ''}
            onClick={() => setActiveTab('protective-factors')}
          >
            Protective Factors
          </button>
        )}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <>
      <div className="summary-stats">
        <div className="stat-card">
          <div className="stat-value">{totalFlags}</div>
          <div className="stat-label">Total Flags</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{flagsBySeverity.high || 0}</div>
          <div className="stat-label">{getSeverityLabel('high')}</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{flagsBySeverity.medium || 0}</div>
          <div className="stat-label">{getSeverityLabel('medium')}</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{flagsBySeverity.low || 0}</div>
          <div className="stat-label">{getSeverityLabel('low')}</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{flagsPerMinute.toFixed(1)}</div>
          <div className="stat-label">Flags/Minute</div>
        </div>
      </div>

      <div className="summary-charts">
        <div className="chart-container">
          <h4>Flags by Severity</h4>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={severityData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {severityData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h4>Top Concern Types</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={typeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name" 
                angle={-45}
                textAnchor="end"
                height={100}
              />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="value" fill="#4a90e2" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      </>
      )}

      {/* Emotion Timeline Tab */}
      {activeTab === 'emotions' && emotionSummary && (
        <EmotionTimeline emotionSummary={emotionSummary} />
      )}

      {/* Evaluation Details Tab */}
      {activeTab === 'evaluations' && evaluations.length > 0 && (
        <div className="evaluation-details">
          <h4>Evaluation Details</h4>
          {evaluations
            .filter(evaluation => {
              // Filter out evaluations with no meaningful content
              const hasFlags = evaluation.flags && Array.isArray(evaluation.flags) && evaluation.flags.length > 0
              const hasMissingQuestions = evaluation.missing_critical_questions && Array.isArray(evaluation.missing_critical_questions) && evaluation.missing_critical_questions.length > 0
              const hasProtectiveFactors = evaluation.protective_factors && Array.isArray(evaluation.protective_factors) && evaluation.protective_factors.length > 0
              return hasFlags || hasMissingQuestions || hasProtectiveFactors
            })
            .map((evaluation, idx) => (
            <div key={idx} className="evaluation-section">
              <h5>Evaluation</h5>
              {evaluation.notes && evaluation.notes.length > 0 && (
                <div className="evaluation-notes">
                  <strong>Notes:</strong>
                  <ul>
                    {evaluation.notes.map((note, noteIdx) => (
                      <li key={noteIdx}>{note}</li>
                    ))}
                  </ul>
                </div>
              )}
              {evaluation.flags && evaluation.flags.length > 0 && (
                <div className="evaluation-flags">
                  <strong>Flags: {evaluation.flags.length}</strong>
                  <div className="flags-list">
                    {evaluation.flags.map((flag, flagIdx) => (
                      <div key={flagIdx} className="evaluation-flag">
                        <span className="flag-time">{flag.timestamp}</span>
                        <span className={`flag-level ${flag.level}`}>{flag.level}</span>
                        <span className="flag-category">{flag.category}</span>
                        {flag.evidence?.quote_or_description && (
                          <p className="flag-quote">"{flag.evidence.quote_or_description}"</p>
                        )}
                        {flag.rationale && (
                          <p className="flag-rationale">{flag.rationale}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Missing Questions Tab */}
      {activeTab === 'missing-questions' && (
        <MissingQuestionsPanel evaluations={evaluations} />
      )}

      {/* Protective Factors Tab */}
      {activeTab === 'protective-factors' && (
        <ProtectiveFactorsPanel 
          evaluations={evaluations} 
          onSeek={onSeek}
        />
      )}
    </div>
  )
}

export default SummaryDashboard
