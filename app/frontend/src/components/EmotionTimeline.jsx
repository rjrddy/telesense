import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './EmotionTimeline.css'

const EmotionTimeline = ({ emotionSummary }) => {
  if (!emotionSummary || !emotionSummary.event_points || emotionSummary.event_points.length === 0) {
    return (
      <div className="emotion-timeline">
        <h4>Emotion Timeline</h4>
        <p className="no-data">No emotion data available</p>
      </div>
    )
  }

  const eventPoints = emotionSummary.event_points || []
  
  // Prepare data for chart - convert to format Recharts expects
  const chartData = eventPoints.map(point => {
    const dataPoint = {
      time: point.time_s || 0,
      timeLabel: point.timestamp || '00:00',
      text: point.evidence?.text || ''
    }
    
    // Add emotion scores
    if (point.emotions) {
      Object.keys(point.emotions).forEach(emotion => {
        dataPoint[emotion] = point.emotions[emotion] || 0
      })
    }
    
    return dataPoint
  })

  // Define colors for each emotion
  const emotionColors = {
    sadness: '#3b82f6',      // Blue
    anxiety: '#ef4444',      // Red
    anger: '#dc2626',        // Dark red
    calm: '#10b981',         // Green
    joy: '#fbbf24',          // Yellow
    pride: '#8b5cf6',         // Purple
    neutral: '#6b7280'        // Gray
  }

  // Get unique emotions from all data points
  const emotions = Object.keys(emotionColors).filter(emotion => 
    chartData.some(point => point[emotion] !== undefined)
  )

  // Format time for display
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="emotion-tooltip">
          <p className="tooltip-time">{formatTime(label)}</p>
          {data.text && <p className="tooltip-text">"{data.text}"</p>}
          <div className="tooltip-emotions">
            {payload
              .filter(item => item.value > 0.1) // Only show emotions with > 10% intensity
              .sort((a, b) => b.value - a.value)
              .map((item, idx) => (
                <div key={idx} className="tooltip-emotion-item">
                  <span className="tooltip-emotion-name">{item.dataKey}:</span>
                  <span className="tooltip-emotion-value">{(item.value * 100).toFixed(1)}%</span>
                </div>
              ))}
          </div>
        </div>
      )
    }
    return null
  }

  return (
    <div className="emotion-timeline">
      <h4>Emotion Timeline</h4>
      
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="time" 
            type="number"
            scale="linear"
            domain={['dataMin', 'dataMax']}
            tickFormatter={formatTime}
            label={{ value: 'Time', position: 'insideBottom', offset: -5 }}
          />
          <YAxis 
            domain={[0, 1]}
            label={{ value: 'Intensity', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          {emotions.map(emotion => (
            <Line
              key={emotion}
              type="monotone"
              dataKey={emotion}
              stroke={emotionColors[emotion]}
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
              name={emotion.charAt(0).toUpperCase() + emotion.slice(1)}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      {/* Display emotion windows if available */}
      {emotionSummary.windows && emotionSummary.windows.length > 0 && (
        <div className="emotion-windows">
          <h5>Emotion Windows</h5>
          <div className="windows-list">
            {emotionSummary.windows.map((window, idx) => (
              <div key={idx} className="emotion-window">
                <div className="window-time">
                  {window.start_timestamp} - {window.end_timestamp}
                </div>
                <div className="window-emotions">
                  {window.dominant_emotions && window.dominant_emotions.map(([emotion, score], i) => (
                    <span key={i} className="window-emotion-tag">
                      {emotion}: {(score * 100).toFixed(0)}%
                    </span>
                  ))}
                </div>
                {window.notes && (
                  <div className="window-notes">{window.notes}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default EmotionTimeline
