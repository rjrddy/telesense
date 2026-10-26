import React, { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import './UploadZone.css'

const UploadZone = ({ onFileUpload }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0]
      // Accept both video and audio files
      const isVideo = file.type.startsWith('video/')
      const isAudio = file.type.startsWith('audio/') || 
                      ['.wav', '.mp3', '.m4a', '.flac'].some(ext => 
                        file.name.toLowerCase().endsWith(ext))
      
      if (isVideo || isAudio) {
        onFileUpload(file)
      } else {
        alert('Please upload a video file (MP4, AVI, MOV, MKV) or audio file (WAV, MP3, M4A, FLAC)')
      }
    }
  }, [onFileUpload])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv'],
      'audio/*': ['.wav', '.mp3', '.m4a', '.flac']
    },
    multiple: false
  })

  return (
    <div className="upload-zone">
      <div {...getRootProps()} className={`upload-dropzone ${isDragActive ? 'active' : ''}`}>
        <input {...getInputProps()} />
        <div className="upload-content">
          <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <h2>{isDragActive ? 'Drop file here' : 'Upload Video or Audio for Analysis'}</h2>
          <p>Drag and drop a video or audio file, or click to select</p>
          <p className="upload-hint">Supported formats: MP4, AVI, MOV, MKV, WAV, MP3, M4A, FLAC</p>
        </div>
      </div>
    </div>
  )
}

export default UploadZone
