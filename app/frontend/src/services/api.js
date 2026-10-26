// API client for backend communication

import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const uploadVideo = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await api.post('/api/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  
  return response.data
}

export const startAnalysis = async (videoId) => {
  const response = await api.post(`/api/analyze/${videoId}`)
  
  return response.data
}

export const getStatus = async (jobId) => {
  const response = await api.get(`/api/status/${jobId}`)
  return response.data
}

export const getResults = async (jobId) => {
  const response = await api.get(`/api/results/${jobId}`)
  return response.data
}

export const createWebSocket = (jobId, onMessage, onError) => {
  const ws = new WebSocket(`ws://localhost:8000/api/ws/${jobId}`)
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data)
    onMessage(data)
  }
  
  ws.onerror = (error) => {
    if (onError) onError(error)
  }

  return ws
}

export default api
