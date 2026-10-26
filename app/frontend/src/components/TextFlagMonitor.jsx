import React, { useRef, useEffect, useMemo, useCallback } from 'react'
import './TextFlagMonitor.css'

const TextFlagMonitor = ({ transcript, flags, currentTime, onSeek }) => {
  const containerRef = useRef(null)
  const scrollTimeoutRef = useRef(null)
  const lastScrolledTimeRef = useRef(-1)

  const allFlags = useMemo(() => (flags && Array.isArray(flags) ? flags : []), [flags])

  // Validate and normalize transcript data
  const normalizedTranscript = useMemo(() => {
    if (!transcript || !Array.isArray(transcript)) {
      return []
    }
    
    return transcript.map((segment, idx) => {
      // Ensure required fields exist
      if (!segment || typeof segment !== 'object') {
        return null
      }
      
      const startTime = typeof segment.start_time === 'number' ? segment.start_time : 0
      const endTime = typeof segment.end_time === 'number' ? segment.end_time : startTime + 1
      const text = segment.text || ''
      
      // Normalize words array
      let words = []
      if (Array.isArray(segment.words) && segment.words.length > 0) {
        words = segment.words
          .filter(word => word && typeof word === 'object' && word.word)
          .map(word => ({
            word: String(word.word),
            start_time: typeof word.start_time === 'number' ? word.start_time : startTime,
            end_time: typeof word.end_time === 'number' ? word.end_time : endTime,
            confidence: typeof word.confidence === 'number' ? word.confidence : null
          }))
      }
      
      return {
        ...segment,
        start_time: startTime,
        end_time: endTime,
        text: String(text),
        words: words,
        index: idx
      }
    }).filter(Boolean) // Remove null entries
  }, [transcript])

  // Validate and normalize flags
  const normalizedFlags = useMemo(() => {
    return allFlags
      .filter(flag => flag && typeof flag === 'object')
      .map(flag => ({
        start_time: typeof flag.start_time === 'number' ? flag.start_time : 0,
        end_time: typeof flag.end_time === 'number' ? flag.end_time : flag.start_time || 0,
        concern_name: String(flag.concern_name || 'Unknown'),
        severity: ['high', 'medium', 'low'].includes(flag.severity) ? flag.severity : 'low',
        confidence: typeof flag.confidence === 'number' ? flag.confidence : 0,
        ...flag
      }))
  }, [allFlags])

  // Improved flag matching with tolerance for timestamp misalignments
  const getFlagsForSegment = useCallback((segment) => {
    if (!segment || !normalizedFlags.length) {
      return []
    }
    
    const tolerance = 0.5 // 0.5 second tolerance for timestamp mismatches
    
    return normalizedFlags.filter(flag => {
      // Check if flag overlaps with segment (with tolerance)
      const flagStart = flag.start_time - tolerance
      const flagEnd = flag.end_time + tolerance
      const segStart = segment.start_time
      const segEnd = segment.end_time
      
      // Overlap conditions:
      // 1. Flag starts within segment
      // 2. Flag ends within segment
      // 3. Flag completely contains segment
      // 4. Segment completely contains flag
      return (
        (flagStart >= segStart && flagStart < segEnd) ||
        (flagEnd > segStart && flagEnd <= segEnd) ||
        (flagStart <= segStart && flagEnd >= segEnd) ||
        (segStart <= flagStart && segEnd >= flagEnd)
      )
    })
  }, [normalizedFlags])

  // Get flags for a specific word
  const getFlagsForWord = useCallback((word, segmentFlags) => {
    if (!word || !segmentFlags.length) {
      return []
    }
    
    const tolerance = 0.2 // Smaller tolerance for word-level matching
    
    return segmentFlags.filter(flag => {
      const flagStart = flag.start_time - tolerance
      const flagEnd = flag.end_time + tolerance
      const wordStart = word.start_time
      const wordEnd = word.end_time
      
      // Word is within flag range or overlaps
      return (
        (wordStart >= flagStart && wordStart < flagEnd) ||
        (wordEnd > flagStart && wordEnd <= flagEnd) ||
        (wordStart <= flagStart && wordEnd >= flagEnd)
      )
    })
  }, [])

  // Improved current segment detection
  const findCurrentSegment = useCallback(() => {
    if (!normalizedTranscript.length || currentTime < 0) {
      return null
    }
    
    // Find segment that contains currentTime
    for (const segment of normalizedTranscript) {
      if (currentTime >= segment.start_time && currentTime < segment.end_time) {
        return segment
      }
    }
    
    // If no exact match, find closest segment
    let closestSegment = null
    let minDistance = Infinity
    
    for (const segment of normalizedTranscript) {
      const distance = Math.min(
        Math.abs(currentTime - segment.start_time),
        Math.abs(currentTime - segment.end_time)
      )
      if (distance < minDistance) {
        minDistance = distance
        closestSegment = segment
      }
    }
    
    return minDistance < 5 ? closestSegment : null // Only return if within 5 seconds
  }, [normalizedTranscript, currentTime])

  // Improved auto-scroll with debouncing
  useEffect(() => {
    if (!containerRef.current || !normalizedTranscript.length) {
      return
    }

    // Clear existing timeout
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current)
    }

    // Debounce scrolling to prevent excessive updates
    scrollTimeoutRef.current = setTimeout(() => {
      const currentSegment = findCurrentSegment()
      
      if (currentSegment && currentSegment.index !== undefined) {
        // Only scroll if time has changed significantly (avoid constant scrolling)
        if (Math.abs(currentTime - lastScrolledTimeRef.current) > 1) {
          const segmentElement = containerRef.current.querySelector(
            `[data-segment-index="${currentSegment.index}"]`
          )
          
          if (segmentElement) {
            segmentElement.scrollIntoView({ 
              behavior: 'smooth', 
              block: 'center',
              inline: 'nearest'
            })
            lastScrolledTimeRef.current = currentTime
          }
        }
      }
    }, 300) // 300ms debounce

    return () => {
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current)
      }
    }
  }, [currentTime, normalizedTranscript, findCurrentSegment])

  const formatTime = (seconds) => {
    if (typeof seconds !== 'number' || isNaN(seconds)) {
      return '0:00'
    }
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const isCurrentSegment = useCallback((segment) => {
    if (!segment || currentTime < 0) {
      return false
    }
    return currentTime >= segment.start_time && currentTime < segment.end_time
  }, [currentTime])

  // Get highest severity for segment (for border color)
  const getHighestSeverity = (segmentFlags) => {
    if (!segmentFlags.length) return null
    
    const severityOrder = { high: 3, medium: 2, low: 1 }  // High Concern reintroduced
    return segmentFlags.reduce((highest, flag) => {
      return severityOrder[flag.severity] > severityOrder[highest.severity] 
        ? flag 
        : highest
    }, segmentFlags[0])
  }

  // Render word with highlighting
  const renderWord = (word, segmentFlags, wordIdx) => {
    const wordFlags = getFlagsForWord(word, segmentFlags)
    const hasFlags = wordFlags.length > 0
    
    return (
      <span
        key={wordIdx}
        className={`word ${hasFlags ? 'flagged' : ''}`}
        title={hasFlags ? wordFlags.map(f => f.concern_name).join(', ') : ''}
      >
        {word.word}{' '}
      </span>
    )
  }

  // Render text when word timestamps are missing
  const renderTextWithoutWords = (text, segmentFlags) => {
    if (!segmentFlags.length) {
      return text
    }
    
    // If we have flags but no word timestamps, highlight the entire text
    // Split text into words for basic highlighting
    const words = text.split(/(\s+)/)
    return words.map((part, idx) => {
      if (part.trim()) {
        return (
          <span key={idx} className="word flagged" title={segmentFlags.map(f => f.concern_name).join(', ')}>
            {part}
          </span>
        )
      }
      return <span key={idx}>{part}</span>
    })
  }

  if (normalizedTranscript.length === 0) {
    return (
      <div className="text-flag-monitor">
        <h3>Text Flag Monitor</h3>
        <div className="transcript-container" ref={containerRef}>
          <div className="no-transcript">No transcript available</div>
        </div>
      </div>
    )
  }

  return (
    <div className="text-flag-monitor">
      <h3>Text Flag Monitor</h3>
      <div className="transcript-container" ref={containerRef}>
        {normalizedTranscript.map((segment) => {
          const segmentFlags = getFlagsForSegment(segment)
          const isCurrent = isCurrentSegment(segment)
          const highestSeverity = getHighestSeverity(segmentFlags)
          const hasFlags = segmentFlags.length > 0
          
          // Check if any flag is emotional_concern
          const hasEmotionalConcern = segmentFlags.some(flag => 
            flag.concern_name && flag.concern_name.toLowerCase().includes('emotional_concern')
          )

          
          // Build class names - use 'low' class for emotional_concern to get yellow color
          const severityClass = hasEmotionalConcern ? 'low' : (highestSeverity ? highestSeverity.severity : '')
          const segmentClasses = [
            'transcript-segment',
            isCurrent ? 'current' : '',
            hasFlags ? 'has-flags' : '',
            severityClass,
            hasEmotionalConcern ? 'emotional-concern' : ''
          ].filter(Boolean).join(' ')
          
          return (
            <div
              key={segment.index}
              data-segment-index={segment.index}
              data-start-time={segment.start_time}
              data-end-time={segment.end_time}
              className={segmentClasses}
              onClick={() => onSeek(segment.start_time)}
            >
              <div className="segment-header">
                <span className="segment-time">
                  [{formatTime(segment.start_time)} - {formatTime(segment.end_time)}]
                </span>
                {hasFlags && (
                  <div className="segment-flags">
                    {segmentFlags.map((flag, flagIdx) => {
                      // Check if it's emotional_concern - always yellow
                      const isEmotionalConcern = flag.concern_name && 
                        flag.concern_name.toLowerCase().includes('emotional_concern')
                      const badgeClass = isEmotionalConcern 
                        ? 'flag-badge low emotional-concern' 
                        : `flag-badge ${flag.severity}`
                      return (
                      <span
                        key={flagIdx}
                        className={badgeClass}
                        title={`${flag.concern_name}: ${(flag.confidence * 100).toFixed(1)}%${flag.rationale ? ` - ${flag.rationale}` : ''}`}
                      >
                        {flag.concern_name}
                      </span>
                      )
                    })}
                  </div>
                )}
              </div>
              <div className="segment-text">
                {segment.words && segment.words.length > 0 ? (
                  segment.words.map((word, wordIdx) => 
                    renderWord(word, segmentFlags, wordIdx)
                  )
                ) : (
                  renderTextWithoutWords(segment.text, segmentFlags)
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default TextFlagMonitor
