import { useEffect, useState, useRef } from 'react'
import { apiService } from '../services/api'

export const useDetectionStatus = (examHall = 1, pollInterval = 1000) => {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const inFlightRef = useRef(false)   // prevent overlapping requests

  useEffect(() => {
    let isMounted = true
    let intervalId

    const fetchStatus = async () => {
      // Skip if a previous request is still in flight — prevents request stacking
      // under load (was causing flickering "OFFLINE" state and stale detections)
      if (inFlightRef.current) return
      inFlightRef.current = true
      try {
        setLoading(true)
        const data = await apiService.getStatus(examHall)
        if (isMounted) {
          setStatus(data)
          setError(null)
        }
      } catch (err) {
        if (isMounted && err.name !== 'AbortError') {
          setError(err.message)
        }
      } finally {
        inFlightRef.current = false
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
