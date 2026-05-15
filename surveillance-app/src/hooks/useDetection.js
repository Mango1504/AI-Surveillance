import { useEffect, useState } from 'react'
import { apiService } from '../services/api'

export const useDetectionStatus = (examHall = 1, pollInterval = 800) => {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let isMounted = true
    let intervalId

    const fetchStatus = async () => {
      try {
        setLoading(true)
        const data = await apiService.getStatus(examHall)
        if (isMounted) {
          setStatus(data)
          setError(null)
        }
      } catch (err) {
        if (isMounted) {
          setError(err.message)
        }
      } finally {
        if (isMounted) {
          setLoading(false)
        }
      }
    }

    fetchStatus()
    intervalId = setInterval(fetchStatus, pollInterval)

    return () => {
      isMounted = false
      clearInterval(intervalId)
    }
  }, [examHall, pollInterval])

  return { status, loading, error }
}

export const useGridInfo = () => {
  const [gridInfo, setGridInfo] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const fetchGridInfo = async () => {
      try {
        setLoading(true)
        const data = await apiService.getGridInfo()
        setGridInfo(data)
      } catch (error) {
        console.error('Failed to fetch grid info:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchGridInfo()
  }, [])

  return { gridInfo, loading }
}
