import { useState, useEffect } from 'react'
import { X, AlertTriangle } from 'lucide-react'
import { useDetectionStatus, useGridInfo } from '../hooks/useDetection'
import useAlertStore from '../context/alertStore'

export default function FeedView({ examHall = 1 }) {
  const [focusedFeed, setFocusedFeed] = useState(null)
  const [streamError, setStreamError] = useState(false)
  const { status, error } = useDetectionStatus(examHall)
  const { gridInfo } = useGridInfo()
  const { addAlert } = useAlertStore()

  // Trigger alert when phone is detected
  useEffect(() => {
    if (status?.phone_detected && status.detections.length > 0) {
      status.detections.forEach((det) => {
        addAlert({
          examHall,
          row: det.grid_row,
          col: det.grid_col,
          confidence: det.confidence,
          timestamp: status.timestamp,
          alert_type: 'Phone Detection',
          clip_path: status.clip_path,
        })
      })
    }
  }, [status?.phone_detected, status?.detections, status?.timestamp, status?.clip_path, examHall, addAlert])

  const renderFeedGrid = () => {
    if (!gridInfo) return null
    const { rows, cols } = gridInfo
    const cells = []
    for (let r = 1; r <= rows; r++) {
      for (let c = 1; c <= cols; c++) {
        cells.push(
          <div
            key={`${r}-${c}`}
            className="grid-cell border border-gray-600 flex items-center justify-center"
          >
            <span className="text-xs text-gray-500">R{r}C{c}</span>
          </div>
        )
      }
    }
    return cells
  }

  const hasDetection = status?.phone_detected && status.detections.length > 0

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Main Feed */}
        <div
          className="col-span-1 md:col-span-2 lg:col-span-3 cursor-pointer transform transition hover:scale-[1.02]"
          onClick={() => setFocusedFeed('main')}
        >
          <div className="bg-black rounded-lg overflow-hidden border border-gray-700 hover:border-gray-500">
            <div className="relative video-feed">
              <img
                src="http://localhost:5000/stream"
                alt={`ExamHall ${examHall} Live Feed`}
                className="w-full h-full object-cover"
                onError={() => setStreamError(true)}
                onLoad={() => setStreamError(false)}
              />
              {streamError && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75">
                  <div className="text-center">
                    <AlertTriangle className="mx-auto mb-2 text-yellow-500" />
                    <p className="text-white">Stream connection error</p>
                    <p className="text-gray-400 text-sm">Retrying...</p>
                  </div>
                </div>
              )}
              {status?.recording && (
                <div className="absolute top-4 right-4">
                  <div className="flex items-center gap-2 bg-red-600 px-3 py-1 rounded-full">
                    <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                    <span className="text-white text-xs font-bold">RECORDING</span>
                  </div>
                </div>
              )}
              {hasDetection && (
                <div className="absolute top-4 left-4">
                  <div className="flex items-center gap-2 bg-red-600 px-3 py-1 rounded-full animate-pulse">
                    <AlertTriangle size={14} className="text-white" />
                    <span className="text-white text-xs font-bold">ALERT</span>
                  </div>
                </div>
              )}
              {gridInfo && (
                <div
                  className="feed-grid-overlay absolute inset-0"
                  style={{
                    gridTemplateRows: `repeat(${gridInfo.rows}, 1fr)`,
                    gridTemplateColumns: `repeat(${gridInfo.cols}, 1fr)`,
                  }}
                >
                  {renderFeedGrid()}
                </div>
              )}
            </div>
          </div>
          <div className="mt-2 p-3 bg-gray-800 rounded-lg">
            <h3 className="font-semibold text-white">ExamHall {examHall}</h3>
            <p className="text-sm text-gray-400">
              {hasDetection ? '🔴 Phone Detected' : '🟢 Normal'}
            </p>
            {status && (
              <p className="text-xs text-gray-500 mt-1">
                {new Date(status.timestamp).toLocaleTimeString()}
              </p>
            )}
          </div>
        </div>

        {/* Detection Info Cards */}
        {status?.detections.map((det, idx) => (
          <div key={idx} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-start justify-between">
              <div>
                <p className="font-semibold text-white text-lg">
                  {det.label}
                </p>
                <p className="text-sm text-gray-400">Detection #{idx + 1}</p>
              </div>
              <div className="text-right">
                <p className="text-sm font-bold text-accent">
                  {(det.confidence * 100).toFixed(0)}%
                </p>
                <p className="text-xs text-gray-500">Confidence</p>
              </div>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
              <div>
                <p className="text-gray-500">Position:</p>
                <p className="text-white font-mono">({det.center[0]}, {det.center[1]})</p>
              </div>
              <div>
                <p className="text-gray-500">Size:</p>
                <p className="text-white font-mono">
                  {det.bbox[2] - det.bbox[0]} × {det.bbox[3] - det.bbox[1]}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {error && (
        <div className="bg-yellow-900 border border-yellow-700 text-yellow-100 px-4 py-3 rounded-lg">
          <p className="font-semibold">Connection Error</p>
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Focused Feed Modal */}
      {focusedFeed && (
        <div className="fixed inset-0 bg-black bg-opacity-95 z-50 flex items-center justify-center p-4">
          <div className="w-full h-full flex flex-col">
            {/* Close button */}
            <button
              onClick={() => setFocusedFeed(null)}
              className="absolute top-4 right-4 bg-red-600 hover:bg-red-700 text-white p-2 rounded-full z-50 transition"
            >
              <X size={24} />
            </button>

            {/* Full screen stream */}
            <div className="flex-1 flex items-center justify-center">
              <img
                src="http://localhost:5000/stream"
                alt="Full Screen Feed"
                className="w-full h-full object-contain"
              />
              {gridInfo && (
                <div
                  className="feed-grid-overlay absolute inset-0"
                  style={{
                    gridTemplateRows: `repeat(${gridInfo.rows}, 1fr)`,
                    gridTemplateColumns: `repeat(${gridInfo.cols}, 1fr)`,
                  }}
                >
                  {renderFeedGrid()}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
