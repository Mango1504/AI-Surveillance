import axios from 'axios'

const API_BASE_URL = 'http://localhost:5000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
})

export const apiService = {
  // Detection endpoints
  getStatus: async (examHall = 1) => {
    try {
      const response = await api.get('/status')
      return { ...response.data, examHall }
    } catch (error) {
      console.error('Error fetching status:', error)
      throw error
    }
  },

  getStreamUrl: () => `${API_BASE_URL}/stream`,

  getGridInfo: async () => {
    try {
      const response = await api.get('/grid-info')
      return response.data
    } catch (error) {
      console.error('Error fetching grid info:', error)
      throw error
    }
  },

  // Fetch real recorded incidents from backend
  getRecords: async () => {
    try {
      const response = await api.get('/incidents')
      return response.data
    } catch (error) {
      console.error('Error fetching records:', error)
      throw error
    }
  },

  // Identified students in current frame
  getIdentifiedStudents: async () => {
    try {
      const response = await api.get('/identified-students')
      return response.data
    } catch (error) {
      console.error('Error fetching identified students:', error)
      return { identified_students: [] }
    }
  },

  // System hardware/performance info
  getSystemInfo: async () => {
    try {
      const response = await api.get('/system-info')
      return response.data
    } catch (error) {
      console.error('Error fetching system info:', error)
      throw error
    }
  },

  // Read current config
  getConfig: async () => {
    try {
      const response = await api.get('/config')
      return response.data
    } catch (error) {
      console.error('Error fetching config:', error)
      throw error
    }
  },

  // Update config
  updateConfig: async (configData) => {
    try {
      const response = await api.post('/config', configData)
      return response.data
    } catch (error) {
      console.error('Error updating config:', error)
      throw error
    }
  },

  // Purge biometric data
  purgeBiometrics: async () => {
    try {
      const response = await api.post('/purge-biometrics')
      return response.data
    } catch (error) {
      console.error('Error purging biometrics:', error)
      throw error
    }
  },

  // Reset session (clear all incidents)
  resetSession: async () => {
    try {
      const response = await api.post('/reset-session')
      return response.data
    } catch (error) {
      console.error('Error resetting session:', error)
      throw error
    }
  },
}

export default api
