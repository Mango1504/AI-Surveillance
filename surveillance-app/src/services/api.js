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

  getSnapshot: async () => {
    try {
      const response = await api.get('/snapshot', { responseType: 'blob' })
      return URL.createObjectURL(response.data)
    } catch (error) {
      console.error('Error fetching snapshot:', error)
      throw error
    }
  },

  getGridInfo: async () => {
    try {
      const response = await api.get('/grid-info')
      return response.data
    } catch (error) {
      console.error('Error fetching grid info:', error)
      throw error
    }
  },

  // Mock video endpoints (replace with actual implementation)
  getVideoClip: async (alertId) => {
    try {
      // This would fetch from the recordings stored by Python code
      // For now returning mock path
      return `/videos/phone_${alertId}.avi`
    } catch (error) {
      console.error('Error fetching video:', error)
      throw error
    }
  },
}

export default api
