import { useEffect, useState } from 'react'
import { AlertTriangle, Shield, RefreshCw } from 'lucide-react'
import io from 'socket.io-client'

// Connect to the Flask-SocketIO backend
const socket = io('http://localhost:5000')

export default function SecurityAlerts() {
  const [alerts, setAlerts] = useState([])
  const [status, setStatus] = useState('CLEAR')

  useEffect(() => {
    // Listen for real-time anomaly detection from VLM backend
    socket.on('anomaly_detected', (data) => {
      setStatus('HIGH')
      setAlerts((prev) => {
        // Keep the latest 10 alerts
        const newAlerts = [data, ...prev].slice(0, 10)
        return newAlerts
      })
    })

    return () => {
      socket.off('anomaly_detected')
    }
  }, [])

  const bgColor = status === 'HIGH' ? 'bg-red-900 border-red-600' : 'bg-green-900 border-green-600'
  const statusColor = status === 'HIGH' ? 'text-red-300' : 'text-green-300'
  const statusBg = status === 'HIGH' ? 'bg-red-600' : 'bg-green-600'

  return (
    <div className={`${bgColor} border-2 rounded-lg p-6 backdrop-blur-sm`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          {status === 'HIGH' ? (
            <AlertTriangle className="w-8 h-8 text-red-400 animate-pulse" />
          ) : (
            <Shield className="w-8 h-8 text-green-400" />
          )}
          <div>
            <h3 className="text-white font-bold text-lg">Security Status</h3>
            <p className="text-gray-300 text-sm">
              {status === 'HIGH' ? 'Active Threat Detected' : 'All Clear'}
            </p>
          </div>
        </div>
        <span className={`${statusBg} text-white font-bold text-lg px-4 py-2 rounded-full`}>
          {status}
        </span>
      </div>

      {alerts.length === 0 ? (
        <div className={`${statusColor} flex items-center gap-2 py-4`}>
          <span>✓</span>
          <span>No security threats detected</span>
        </div>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {alerts.map((alert, idx) => (
            <div
              key={idx}
              className="bg-red-950 bg-opacity-50 border border-red-500 p-4 rounded-lg hover:bg-opacity-70 transition"
            >
              <div className="flex items-start justify-between mb-2">
                <p className="text-red-200 font-semibold text-base">
                  ⚠️ {alert.report}
                </p>
                <span className="bg-red-600 text-white text-xs px-2 py-1 rounded">
                  {alert.labels ? alert.labels.join(', ') : 'anomaly'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs text-red-300">
                <p>Candidate ID: {alert.candidate_id}</p>
                <p>Time: {new Date(alert.timestamp * 1000).toLocaleTimeString()}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-4 pt-4 border-t border-gray-700 text-xs text-gray-400">
        <p>Total Incidents Logged: <span className="text-red-300 font-bold">{alerts.length}</span></p>
      </div>
    </div>
  )
}
