import React, { useRef, useEffect, useState, useCallback } from 'react'
import WaveSurfer from 'wavesurfer.js'
import './AudioFlagMonitor.css'

const AudioFlagMonitor = ({ flags, duration, currentTime, onSeek, videoUrl }) => {
  const waveformRef = useRef(null)
  const wavesurferRef = useRef(null)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState(null)
  const flagsRef = useRef([])
  const isProgrammaticSeekRef = useRef(false)

  // Keep only well-formed flags
  const normalizedFlags = React.useMemo(() => {
    return flags && Array.isArray(flags) ? flags.filter(flag => 
      flag && 
      typeof flag === 'object' && 
      typeof flag.start_time === 'number' && 
      typeof flag.end_time === 'number'
    ) : []
  }, [flags])

  // Initialize WaveSurfer
  useEffect(() => {
    if (!waveformRef.current) return

    let wavesurfer = null

    try {
      // Create WaveSurfer instance
      wavesurfer = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#4a90e2',
        progressColor: '#3b82f6',
        cursorColor: '#ef4444',
        barWidth: 2,
        barRadius: 2,
        responsive: true,
        height: 100,
        normalize: true,
        backend: 'WebAudio',
        mediaControls: false,
        interact: true,  // Allow user interaction
        hideScrollbar: false,
      })

      wavesurferRef.current = wavesurfer

      // Handle ready event
      const handleReady = () => {
        setIsReady(true)
        setError(null)
      }

      // Handle errors
      const handleError = (err) => {
        console.error('WaveSurfer error:', err)
        setError('Failed to load audio waveform')
        setIsReady(false)
      }

      wavesurfer.on('ready', handleReady)
      wavesurfer.on('error', handleError)

      // Load audio if videoUrl is provided
      if (videoUrl) {
        // Check if it's an audio URL (ends with .wav, .mp3, etc.) or contains /audio/
        const isAudioUrl = /\.(wav|mp3|m4a|flac|ogg)$/i.test(videoUrl) || 
                          /\/audio\//i.test(videoUrl) ||
                          videoUrl.includes('audio.wav')
        
        if (isAudioUrl) {
          // Direct audio URL - load it
          wavesurfer.load(videoUrl)
            .then(() => {
              // Successfully loaded
              setError(null)
            })
            .catch((err) => {
              console.error('Failed to load audio URL:', err)
              setError('Failed to load audio file. Please check if the audio URL is accessible.')
            })
        } else {
          // Video URL or unknown format - WaveSurfer can't load video directly
          // Don't try to load, just show a message that audio URL is needed
          setError(null) // Don't show error, waveform just won't display
          console.info('Audio waveform requires audio URL. Video URLs are not supported by WaveSurfer.')
        }
      } else {
        // No URL provided - don't show error, just don't display waveform
        setError(null)
      }

      // Handle seek events (only if user-initiated, not programmatic)
      let isUserSeeking = false
      
      // Track user interactions
      const handleInteraction = () => {
        isUserSeeking = true
      }
      
      // Listen for clicks/drags on the waveform
      if (waveformRef.current) {
        waveformRef.current.addEventListener('mousedown', handleInteraction)
        waveformRef.current.addEventListener('touchstart', handleInteraction)
      }
      
      wavesurfer.on('seek', (progress) => {
        // Only trigger onSeek if it was a user interaction, not programmatic
        if (isUserSeeking && !isProgrammaticSeekRef.current && duration > 0) {
          const time = progress * duration
          onSeek(time)
        }
        isUserSeeking = false
      })
      
      // Store the interaction handler for cleanup
      wavesurfer._interactionHandler = handleInteraction

    } catch (err) {
      console.error('Error creating WaveSurfer:', err)
      setError('Failed to initialize waveform')
    }

    return () => {
      if (wavesurfer) {
        try {
          wavesurfer.destroy()
        } catch (err) {
          console.error('Error destroying WaveSurfer:', err)
        }
      }
      wavesurferRef.current = null
      setIsReady(false)
    }
  }, [videoUrl, duration, onSeek])

  // Sync current time with debouncing to prevent flickering
  const lastSeekTimeRef = useRef(-1)
  const lastProgressRef = useRef(-1)
  const seekTimeoutRef = useRef(null)
  
  useEffect(() => {
    if (!wavesurferRef.current || !isReady || duration <= 0 || currentTime < 0) {
      return
    }

    // Clear existing timeout
    if (seekTimeoutRef.current) {
      clearTimeout(seekTimeoutRef.current)
    }

    // Only seek if time changed significantly (more than 0.2 seconds)
    const timeDiff = Math.abs(currentTime - lastSeekTimeRef.current)
    if (timeDiff < 0.2) {
      return
    }

    // Debounce seek operations to prevent flickering
    seekTimeoutRef.current = setTimeout(() => {
      if (wavesurferRef.current && isReady) {
        try {
          const progress = Math.max(0, Math.min(1, currentTime / duration))
          // Only seek if progress changed significantly (more than 1%)
          if (Math.abs(progress - lastProgressRef.current) > 0.01) {
            // Mark as programmatic seek to prevent callback loop
            isProgrammaticSeekRef.current = true
            wavesurferRef.current.seekTo(progress)
            lastSeekTimeRef.current = currentTime
            lastProgressRef.current = progress
            // Reset flag after a short delay
            setTimeout(() => {
              isProgrammaticSeekRef.current = false
            }, 50)
          }
        } catch (err) {
          console.error('Error seeking waveform:', err)
          isProgrammaticSeekRef.current = false
        }
      }
    }, 150) // 150ms debounce to reduce flickering

    return () => {
      if (seekTimeoutRef.current) {
        clearTimeout(seekTimeoutRef.current)
      }
    }
  }, [currentTime, duration, isReady])

  // Draw flags on waveform
  useEffect(() => {
    if (!waveformRef.current || !isReady || normalizedFlags.length === 0) {
      // Clear flags if not ready
      if (waveformRef.current) {
        waveformRef.current.querySelectorAll('.audio-flag-marker').forEach(el => el.remove())
      }
      return
    }

    const container = waveformRef.current
    if (!container) return

    // Remove existing flag markers
    container.querySelectorAll('.audio-flag-marker').forEach(el => el.remove())

    // Store flags for reference
    flagsRef.current = normalizedFlags

    // Create flag markers
    normalizedFlags.forEach((flag, idx) => {
      try {
        const marker = document.createElement('div')
        // Check if it's emotional_concern - always yellow
        const isEmotionalConcern = flag.concern_name && 
          flag.concern_name.toLowerCase().includes('emotional_concern')
        const severityClass = flag.severity || 'low'
        marker.className = `audio-flag-marker ${severityClass} ${isEmotionalConcern ? 'emotional-concern' : ''}`
        
        if (duration > 0) {
          const startPercent = Math.max(0, Math.min(100, (flag.start_time / duration) * 100))
          const endPercent = Math.max(0, Math.min(100, (flag.end_time / duration) * 100))
          const widthPercent = Math.max(1, endPercent - startPercent)
          
          marker.style.position = 'absolute'
          marker.style.left = `${startPercent}%`
          marker.style.width = `${widthPercent}%`
          marker.style.top = '0'
          marker.style.height = '100%'
          marker.style.pointerEvents = 'auto'
          marker.style.cursor = 'pointer'
          marker.style.zIndex = '10'
        }
        
        const concernName = flag.concern_name || 'Concern'
        const confidence = typeof flag.confidence === 'number' ? flag.confidence : 0
        const severityLabel = flag.severity === 'high' ? 'High Risk' : (flag.severity === 'medium' ? 'High Concern' : 'Concern')
        marker.title = `${concernName}: ${(confidence * 100).toFixed(1)}% (${severityLabel})`
        
        marker.addEventListener('click', (e) => {
          e.stopPropagation()
          if (typeof flag.start_time === 'number') {
            onSeek(flag.start_time)
          }
        })
        
        container.appendChild(marker)
      } catch (err) {
        console.error('Error creating flag marker:', err)
      }
    })
  }, [normalizedFlags, duration, isReady, onSeek])

  return (
    <div className="audio-flag-monitor">
      <h3>Audio Flag Monitor</h3>
      <div className="monitor-legend">
        <span className="legend-item high">High Risk</span>
        <span className="legend-item medium">High Concern</span>
        <span className="legend-item low">Concern</span>
      </div>
      <div className="waveform-wrapper">
        <div ref={waveformRef} className="waveform-container"></div>
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
        {!isReady && !error && videoUrl && (
          <div className="loading-message">Loading audio waveform...</div>
        )}
        {!videoUrl && !error && (
          <div className="loading-message" style={{ color: '#6b7280' }}>Audio waveform will appear when audio is available</div>
        )}
      </div>
    </div>
  )
}

export default AudioFlagMonitor
