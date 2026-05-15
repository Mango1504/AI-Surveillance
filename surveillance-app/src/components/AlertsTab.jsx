import { useState } from 'react'
import { Trash2, PlayCircle, X } from 'lucide-react'
import useAlertStore from '../context/alertStore'

export default function AlertsTab() {
  const { alerts, deleteAlert } = useAlertStore()
  const [selectedAlert, setSelectedAlert] = useState(null)
  const [videoModalOpen, setVideoModalOpen] = useState(false)

  const handlePlayVideo = (alert) => {
    setSelectedAlert(alert)
    setVideoModalOpen(true)
  }

  const handleDeleteAlert = (id) => {
    deleteAlert(id)
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">Alerts ({alerts.length})</h2>
      </div>

      {alerts.length === 0 ? (
        <div className="bg-gray-800 rounded-lg p-12 text-center border border-gray-700">
          <div className="text-gray-400 text-lg">No alerts yet</div>
          <p className="text-gray-500 text-sm mt-1">
            Phone detections will appear here
          </p>
        </div>
      ) : (
        <div className="grid gap-4">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className="bg-gray-800 rounded-lg p-4 border-l-4 border-red-500 hover:bg-gray-750 transition"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="bg-red-600 text-white px-2 py-1 rounded text-xs font-bold">
                      {alert.alert_type}
                    </span>
                    <span className="text-gray-400 text-sm">
                      ExamHall {alert.examHall}
                    </span>
                  </div>
                  <p className="text-white font-semibold mb-2">
                    Detection at Row {alert.row}, Column {alert.col}
                  </p>
                  <div className="grid grid-cols-2 gap-4 text-sm mb-3">
                    <div>
                      <p className="text-gray-500">Confidence:</p>
                      <p className="text-accent font-mono">{(alert.confidence * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Time:</p>
                      <p className="text-white font-mono">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex gap-2 ml-4">
                  {alert.clip_path && (
                    <button
                      onClick={() => handlePlayVideo(alert)}
                      className="flex items-center gap-1 px-3 py-2 bg-primary hover:bg-blue-700 text-white rounded transition text-sm"
                    >
                      <PlayCircle size={16} />
                      <span>Video</span>
                    </button>
                  )}
                  <button
                    onClick={() => handleDeleteAlert(alert.id)}
                    className="flex items-center gap-1 px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded transition"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Video Modal */}
      {videoModalOpen && selectedAlert && (
        <div className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center p-4">
          <div className="bg-gray-800 rounded-lg max-w-4xl w-full overflow-hidden">
            {/* Header */}
            <div className="flex justify-between items-center p-4 border-b border-gray-700">
              <div>
                <h3 className="text-lg font-bold text-white">
                  {selectedAlert.alert_type}
                </h3>
                <p className="text-sm text-gray-400">
                  Row {selectedAlert.row}, Col {selectedAlert.col} • ExamHall {selectedAlert.examHall}
                </p>
              </div>
              <button
                onClick={() => setVideoModalOpen(false)}
                className="p-2 hover:bg-gray-700 rounded transition"
              >
                <X size={24} className="text-white" />
              </button>
            </div>

            {/* Video Player (MJPEG stream) */}
            <div className="bg-black aspect-video flex items-center justify-center overflow-hidden">
              {selectedAlert.clip_path ? (
                <img
                  src={selectedAlert.clip_path}
                  alt="Alert recording playback"
                  className="w-full h-full object-contain"
                />
              ) : (
                <p className="text-gray-500">No video available for this alert</p>
              )}
            </div>

            {/* Details */}
            <div className="p-4 bg-gray-750 grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-gray-500">Confidence</p>
                <p className="text-white font-semibold">{(selectedAlert.confidence * 100).toFixed(1)}%</p>
              </div>
              <div>
                <p className="text-gray-500">Timestamp</p>
                <p className="text-white font-semibold">
                  {new Date(selectedAlert.timestamp).toLocaleTimeString()}
                </p>
              </div>
              <div>
                <p className="text-gray-500">Duration</p>
                <p className="text-white font-semibold">5 seconds (min)</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
