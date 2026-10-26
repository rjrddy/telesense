import React, { useState, useEffect, useRef } from 'react'
import UploadZone from './components/UploadZone'
import VideoPlayer from './components/VideoPlayer'
import VisualFlagMonitor from './components/VisualFlagMonitor'
import AudioFlagMonitor from './components/AudioFlagMonitor'
import TextFlagMonitor from './components/TextFlagMonitor'
import SummaryDashboard from './components/SummaryDashboard'
import { uploadVideo, startAnalysis, getStatus, getResults, createWebSocket } from './services/api'
import './styles/App.css'

function App() {
  const [videoFile, setVideoFile] = useState(null)
  const [videoId, setVideoId] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [analysisResults, setAnalysisResults] = useState(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [processingStatus, setProcessingStatus] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [fileType, setFileType] = useState(null) // 'video' or 'audio'
  const wsRef = useRef(null)

  // Handle file upload
  const handleFileUpload = async (file) => {
    try {
      setVideoFile(file)
      const result = await uploadVideo(file)
      setVideoId(result.video_id)
      
      // Determine file type
      const isAudio = file.type.startsWith('audio/') || 
                      ['.wav', '.mp3', '.m4a', '.flac'].some(ext => 
                        file.name.toLowerCase().endsWith(ext))
      setFileType(isAudio ? 'audio' : 'video')
      
      // Create object URL for preview
      const url = URL.createObjectURL(file)
      setVideoUrl(url)
      
      // Start analysis
      const analysisJob = await startAnalysis(result.video_id)
      setJobId(analysisJob.job_id)
      setProcessingStatus(analysisJob)
      
      // Connect WebSocket for updates
      if (wsRef.current) {
        wsRef.current.close()
      }
      
      wsRef.current = createWebSocket(
        analysisJob.job_id,
        (data) => {
          setProcessingStatus(data)
          if (data.status === 'completed') {
            loadResults(analysisJob.job_id)
          }
        },
        (error) => {
          console.error('WebSocket error:', error)
        }
      )
    } catch (error) {
      console.error('Error uploading file:', error)
      alert('Error uploading file: ' + error.message)
    }
  }

  // Load analysis results
  const loadResults = async (jobId) => {
    try {
      const results = await getResults(jobId)
      setAnalysisResults(results)
      // Update file type from results if available
      if (results && results.file_type) {
        setFileType(results.file_type)
      }
    } catch (error) {
      console.error('Error loading results:', error)
      // Retry after a short delay if still processing
      if (error.response?.status === 202) {
        setTimeout(() => loadResults(jobId), 2000)
      }
    }
  }

  // Poll for status if WebSocket fails
  useEffect(() => {
    if (jobId && processingStatus?.status === 'processing') {
      const interval = setInterval(async () => {
        try {
          const status = await getStatus(jobId)
          setProcessingStatus(status)
          if (status.status === 'completed') {
            clearInterval(interval)
            loadResults(jobId)
          } else if (status.status === 'failed') {
            clearInterval(interval)
          }
        } catch (error) {
          console.error('Error polling status:', error)
        }
      }, 2000)
      
      return () => clearInterval(interval)
    }
  }, [jobId, processingStatus])

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  const handleSeek = (time) => {
    setCurrentTime(time)
  }

  const handleTimeUpdate = (time) => {
    setCurrentTime(time)
  }

  const handlePlayPause = (playing) => {
    setIsPlaying(playing)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Telesense - Post-intake Multi-Modal Fusion Analysis</h1>
        <div className="header-actions">
          {analysisResults && (
            <button 
              onClick={() => {
                const dataStr = JSON.stringify(analysisResults, null, 2)
                const dataBlob = new Blob([dataStr], { type: 'application/json' })
                const url = URL.createObjectURL(dataBlob)
                const link = document.createElement('a')
                link.href = url
                link.download = `analysis_${jobId}.json`
                link.click()
              }}
              className="export-button"
            >
              Export JSON
            </button>
          )}
        </div>
      </header>

      <main className="app-main">
        {!videoFile ? (
          <UploadZone onFileUpload={handleFileUpload} />
        ) : (
          <>
            {processingStatus && processingStatus.status !== 'completed' && (
              <div className="processing-status">
                <div className="status-bar">
                  <div 
                    className="status-progress" 
                    style={{ width: `${processingStatus.progress * 100}%` }}
                  />
                </div>
                <p>{processingStatus.message} ({Math.round(processingStatus.progress * 100)}%)</p>
              </div>
            )}

            <div className="video-section">
              <VideoPlayer
                videoUrl={fileType === 'audio' && analysisResults?.audio_url 
                  ? `http://localhost:8000${analysisResults.audio_url}` 
                  : videoUrl}
                currentTime={currentTime}
                onTimeUpdate={handleTimeUpdate}
                onSeek={handleSeek}
                onPlayPause={handlePlayPause}
                flags={analysisResults?.combined_flags || []}
                isAudio={fileType === 'audio'}
              />
            </div>

            <div className="monitors-section">
              <VisualFlagMonitor
                flags={analysisResults?.visual_flags || []}
                duration={analysisResults?.duration || 0}
                currentTime={currentTime}
                onSeek={handleSeek}
              />

              <AudioFlagMonitor
                flags={analysisResults?.audio_flags || []}
                duration={analysisResults?.duration || 0}
                currentTime={currentTime}
                onSeek={handleSeek}
                videoUrl={analysisResults?.audio_url ? `http://localhost:8000${analysisResults.audio_url}` : videoUrl}
              />

              <TextFlagMonitor
                transcript={analysisResults?.transcript || []}
                flags={analysisResults?.text_flags || []}
                currentTime={currentTime}
                onSeek={handleSeek}
              />
            </div>

            {analysisResults && (
              <SummaryDashboard
                results={analysisResults}
                onSeek={handleSeek}
              />
            )}
          </>
        )}
      </main>
    </div>
  )
}

export default App
