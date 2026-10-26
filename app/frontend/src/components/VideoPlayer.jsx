import React, { useRef, useEffect, useState } from 'react'
import './VideoPlayer.css'

const VideoPlayer = ({ videoUrl, currentTime, onTimeUpdate, onSeek, onPlayPause, flags, isAudio = false }) => {
  const videoRef = useRef(null)
  const audioRef = useRef(null)
  const mediaRef = isAudio ? audioRef : videoRef
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1)
  const isInternalUpdateRef = useRef(false) // Track if update is from video's own timeupdate
  const lastSyncedTimeRef = useRef(-1) // Track last synced time to prevent unnecessary updates
  const syncTimeoutRef = useRef(null) // Debounce sync operations

  // Setup event listeners (only once, not on every currentTime change)
  useEffect(() => {
    const media = mediaRef.current
    if (!media) return

    const handleTimeUpdate = () => {
      // Mark as internal update to prevent sync loop
      isInternalUpdateRef.current = true
      onTimeUpdate(media.currentTime)
      lastSyncedTimeRef.current = media.currentTime
      // Reset flag after a short delay
      setTimeout(() => {
        isInternalUpdateRef.current = false
      }, 100)
    }
    
    const handleLoadedMetadata = () => {
      setDuration(media.duration)
    }
    
    const handlePlay = () => {
      setIsPlaying(true)
      onPlayPause(true)
    }
    
    const handlePause = () => {
      setIsPlaying(false)
      onPlayPause(false)
    }
    
    media.addEventListener('timeupdate', handleTimeUpdate)
    media.addEventListener('loadedmetadata', handleLoadedMetadata)
    media.addEventListener('play', handlePlay)
    media.addEventListener('pause', handlePause)
    
    return () => {
      media.removeEventListener('timeupdate', handleTimeUpdate)
      media.removeEventListener('loadedmetadata', handleLoadedMetadata)
      media.removeEventListener('play', handlePlay)
      media.removeEventListener('pause', handlePause)
    }
  }, [videoUrl, onTimeUpdate, onPlayPause, mediaRef]) // Only re-run if videoUrl changes

  // Separate effect to sync external currentTime changes (only when needed)
  useEffect(() => {
    // Clear any pending sync
    if (syncTimeoutRef.current) {
      clearTimeout(syncTimeoutRef.current)
    }

    const media = mediaRef.current
    if (!media || isInternalUpdateRef.current) {
      // Don't sync if update is from video's own timeupdate
      return
    }

    // Debounce sync to prevent rapid updates
    syncTimeoutRef.current = setTimeout(() => {
      if (!media || isInternalUpdateRef.current) {
        return
      }

      // Calculate time difference
      const timeDiff = Math.abs(media.currentTime - currentTime)
      
      // When video is playing, don't sync at all (let it play naturally)
      // Only sync when paused or for very large differences (seek from another component)
      if (isPlaying) {
        // When playing, only sync for very large differences (5+ seconds) to avoid interrupting playback
        if (timeDiff > 5.0) {
          media.currentTime = currentTime
          lastSyncedTimeRef.current = currentTime
        }
      } else {
        // When paused, sync if difference is significant (more than 0.5 seconds)
        if (timeDiff > 0.5) {
          media.currentTime = currentTime
          lastSyncedTimeRef.current = currentTime
        }
      }
    }, 200) // 200ms debounce

    return () => {
      if (syncTimeoutRef.current) {
        clearTimeout(syncTimeoutRef.current)
      }
    }
  }, [currentTime, isPlaying]) // Only depend on currentTime and isPlaying

  const togglePlayPause = () => {
    const media = mediaRef.current
    if (media) {
      if (isPlaying) {
        media.pause()
      } else {
        media.play()
      }
    }
  }

  const handleSeek = (e) => {
    const media = mediaRef.current
    if (media) {
      const rect = e.currentTarget.getBoundingClientRect()
      const x = e.clientX - rect.left
      const percent = x / rect.width
      const newTime = percent * duration
      
      // Mark as internal update to prevent sync loop
      isInternalUpdateRef.current = true
      media.currentTime = newTime
      lastSyncedTimeRef.current = newTime
      onSeek(newTime)
      
      // Reset flag after a short delay
      setTimeout(() => {
        isInternalUpdateRef.current = false
      }, 100)
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getFlagAtTime = (time) => {
    return flags.find(flag => 
      flag.start_time <= time && time <= flag.end_time
    )
  }

  const currentFlag = getFlagAtTime(currentTime)

  return (
    <div className="video-player-container">
      <div className={`video-wrapper ${isAudio ? 'audio-wrapper' : ''}`}>
        {isAudio ? (
          <div className="audio-visualization">
            <audio
              ref={audioRef}
              src={videoUrl}
              className="audio-element"
            />
            <div className="audio-waveform-placeholder">
              <svg viewBox="0 0 400 200" className="waveform-svg">
                <path
                  d="M 0 100 Q 50 50, 100 100 T 200 100 T 300 100 T 400 100"
                  stroke="currentColor"
                  strokeWidth="2"
                  fill="none"
                />
              </svg>
              <p>Audio Player</p>
            </div>
          </div>
        ) : (
          <video
            ref={videoRef}
            src={videoUrl}
            className="video-element"
            onClick={togglePlayPause}
            preload="metadata"
          />
        )}
        
        {currentFlag && (() => {
          // Check if it's emotional_concern - always yellow
          const isEmotionalConcern = currentFlag.concern_name && 
            currentFlag.concern_name.toLowerCase().includes('emotional_concern')
          const overlayClass = isEmotionalConcern ? 'video-flag-overlay low emotional-concern' : `video-flag-overlay ${currentFlag.severity}`
          return (
            <div className={overlayClass}>
              <span>{currentFlag.concern_name}</span>
            </div>
          )
        })()}
      </div>

      <div className="video-controls">
        <button className="play-pause-button" onClick={togglePlayPause}>
          {isPlaying ? '⏸' : '▶'}
        </button>

        <div className="timeline-container" onClick={handleSeek}>
          <div className="timeline-track">
            <div 
              className="timeline-progress" 
              style={{ width: `${(currentTime / duration) * 100}%` }}
            />
            {flags.map((flag, idx) => {
              // Check if it's emotional_concern - always yellow
              const isEmotionalConcern = flag.concern_name && 
                flag.concern_name.toLowerCase().includes('emotional_concern')
              const flagClass = isEmotionalConcern ? 'timeline-flag low emotional-concern' : `timeline-flag ${flag.severity}`
              return (
                <div
                  key={idx}
                  className={flagClass}
                  style={{
                    left: `${(flag.start_time / duration) * 100}%`,
                    width: `${((flag.end_time - flag.start_time) / duration) * 100}%`
                  }}
                  title={`${flag.concern_name} (${flag.confidence.toFixed(2)})`}
                />
              )
            })}
          </div>
        </div>

        <div className="time-display">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>

        <div className="volume-control">
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={volume}
            onChange={(e) => {
              const newVolume = parseFloat(e.target.value)
              setVolume(newVolume)
              const media = mediaRef.current
              if (media) {
                media.volume = newVolume
              }
            }}
          />
        </div>
      </div>
    </div>
  )
}

export default VideoPlayer
