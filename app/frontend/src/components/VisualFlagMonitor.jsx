import React, { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import './VisualFlagMonitor.css'

const VisualFlagMonitor = ({ flags, duration, currentTime, onSeek }) => {
  const svgRef = useRef(null)
  const containerRef = useRef(null)

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || duration === 0) return

    const width = containerRef.current.offsetWidth
    const height = 60
    const margin = { top: 10, right: 10, bottom: 30, left: 10 }

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)

    const xScale = d3.scaleLinear()
      .domain([0, duration])
      .range([margin.left, width - margin.right])

    // Draw timeline background
    svg.append('rect')
      .attr('x', margin.left)
      .attr('y', margin.top)
      .attr('width', width - margin.left - margin.right)
      .attr('height', height - margin.top - margin.bottom)
      .attr('fill', '#f0f0f0')
      .attr('rx', 4)

    // Draw flags
    flags.forEach(flag => {
      const x = xScale(flag.start_time)
      const width_flag = xScale(flag.end_time) - xScale(flag.start_time)
      
      let color
      // Check if it's emotional_concern - always yellow
      const isEmotionalConcern = flag.concern_name && 
        flag.concern_name.toLowerCase().includes('emotional_concern')
      
      if (isEmotionalConcern) {
        color = '#fbbf24'  // Yellow for emotional_concern
      } else if (flag.severity === 'high') {
        color = '#ef4444'  // Red for high risk
      } else if (flag.severity === 'medium') {
        color = '#f59e0b'  // Orange for high concern
      } else {
        color = '#fbbf24'  // Yellow for concern (low)
      }

      svg.append('rect')
        .attr('x', x)
        .attr('y', margin.top)
        .attr('width', Math.max(width_flag, 2))
        .attr('height', height - margin.top - margin.bottom)
        .attr('fill', color)
        .attr('opacity', 0.7)
        .attr('rx', 2)
        .style('cursor', 'pointer')
        .on('click', () => onSeek(flag.start_time))
        .append('title')
        .text(`${flag.concern_name}: ${(flag.confidence * 100).toFixed(1)}%`)
    })

    // Draw current time indicator
    const currentX = xScale(currentTime)
    svg.append('line')
      .attr('x1', currentX)
      .attr('x2', currentX)
      .attr('y1', 0)
      .attr('y2', height)
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2)
      .attr('pointer-events', 'none')

    // Draw time scale
    const timeAxis = d3.axisBottom(xScale)
      .ticks(10)
      .tickFormat(d => {
        const mins = Math.floor(d / 60)
        const secs = Math.floor(d % 60)
        return `${mins}:${secs.toString().padStart(2, '0')}`
      })

    svg.append('g')
      .attr('transform', `translate(0, ${height - margin.bottom})`)
      .call(timeAxis)
      .selectAll('text')
      .style('font-size', '10px')

    // Make timeline clickable
    svg.append('rect')
      .attr('x', margin.left)
      .attr('y', margin.top)
      .attr('width', width - margin.left - margin.right)
      .attr('height', height - margin.top - margin.bottom)
      .attr('fill', 'transparent')
      .style('cursor', 'pointer')
      .on('click', (event) => {
        const [x] = d3.pointer(event)
        const time = xScale.invert(x)
        onSeek(Math.max(0, Math.min(time, duration)))
      })

  }, [flags, duration, currentTime, onSeek])

  return (
    <div className="visual-flag-monitor">
      <h3>Visual Flag Monitor</h3>
      <div className="monitor-legend">
        <span className="legend-item high">High Risk</span>
        <span className="legend-item medium">High Concern</span>
        <span className="legend-item low">Concern</span>
      </div>
      <div ref={containerRef} className="timeline-svg-container">
        <svg ref={svgRef}></svg>
      </div>
    </div>
  )
}

export default VisualFlagMonitor
