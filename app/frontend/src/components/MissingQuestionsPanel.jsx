import React from 'react'
import './MissingQuestionsPanel.css'

const MissingQuestionsPanel = ({ evaluations }) => {
  if (!evaluations || evaluations.length === 0) {
    return (
      <div className="missing-questions-panel">
        <h4>Missing Critical Questions</h4>
        <p className="no-data">No evaluation data available</p>
      </div>
    )
  }

  // Collect all missing questions from all evaluations
  const allQuestions = []
  evaluations.forEach((evaluation, evalIdx) => {
    const missingQuestions = evaluation.missing_critical_questions || []
    if (missingQuestions.length > 0) {
      allQuestions.push({
        evaluationName: evaluation.name,
        evaluationBasis: evaluation.basis,
        questions: missingQuestions
      })
    }
  })

  if (allQuestions.length === 0) {
    return (
      <div className="missing-questions-panel">
        <h4>Missing Critical Questions</h4>
        <p className="no-data">No missing critical questions identified</p>
      </div>
    )
  }

  // Categorize questions
  const categorizeQuestion = (question) => {
    const qLower = question.toLowerCase()
    if (qLower.includes('si') || qLower.includes('suicide') || qLower.includes('homicide') || qLower.includes('hi') || qLower.includes('safety')) {
      return 'Safety'
    }
    if (qLower.includes('psychosis') || qLower.includes('mania') || qLower.includes('substance')) {
      return 'Clinical'
    }
    if (qLower.includes('sleep') || qLower.includes('trauma') || qLower.includes('depression') || qLower.includes('anxiety')) {
      return 'Assessment'
    }
    if (qLower.includes('functional') || qLower.includes('social') || qLower.includes('support')) {
      return 'Functional'
    }
    return 'Other'
  }

  return (
    <div className="missing-questions-panel">
      <h4>Missing Critical Questions</h4>
      <p className="panel-description">
        Questions that should be asked to complete the clinical assessment based on available data.
      </p>

      {allQuestions.map((evalData, idx) => (
        <div key={idx} className="evaluation-questions">
          <h5 className="evaluation-title">
            Evaluation
            <span className="evaluation-basis">({evalData.evaluationBasis})</span>
          </h5>
          
          {/* Group questions by category */}
          {(() => {
            const categorized = {}
            evalData.questions.forEach(q => {
              const category = categorizeQuestion(q)
              if (!categorized[category]) {
                categorized[category] = []
              }
              categorized[category].push(q)
            })

            return Object.entries(categorized).map(([category, questions]) => (
              <div key={category} className="question-category">
                <h6 className="category-header">
                  <span className="category-icon">{getCategoryIcon(category)}</span>
                  {category}
                  <span className="question-count">({questions.length})</span>
                </h6>
                <ul className="questions-list">
                  {questions.map((question, qIdx) => (
                    <li key={qIdx} className="question-item">
                      <span className="question-icon">❓</span>
                      <span className="question-text">{question}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))
          })()}
        </div>
      ))}
    </div>
  )
}

// Helper function to get icon for category
const getCategoryIcon = (category) => {
  const icons = {
    'Safety': '⚠️',
    'Clinical': '🏥',
    'Assessment': '📋',
    'Functional': '⚙️',
    'Other': '📝'
  }
  return icons[category] || '📝'
}

export default MissingQuestionsPanel
