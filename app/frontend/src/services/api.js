// API client for backend communication

import axios from 'axios'

const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

// WebSocket URL derived from the API URL: http(s)://host -> ws(s)://host
const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws')

// Attach the demo gate key if set via env, or if the user entered one locally.
function getDemoKey() {
  try {
    return (
      import.meta.env.VITE_TELESENSE_DEMO_KEY ||
      localStorage.getItem('telesense_demo_key') ||
      ''
    )
  } catch {
    return ''
  }
}

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use((config) => {
  const key = getDemoKey()
  if (key) config.headers['X-Telesense-Key'] = key
  return config
})

export const authUrl = (path) => {
  const url = new URL(`${API_BASE_URL}${path}`)
  const key = getDemoKey()
  if (key) url.searchParams.set('key', key)
  return url.toString()
}

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
  const key = getDemoKey()
  const qs = key ? `?key=${encodeURIComponent(key)}` : ''
  const ws = new WebSocket(`${WS_BASE_URL}/api/ws/${jobId}${qs}`)

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
