import { useState, useEffect } from 'react'
import { PlayCircle, X, RefreshCw } from 'lucide-react'
import { apiService } from '../services/api'

export default function RecordsView() {
  const [records, setRecords] = useState([])
  const [videoModal, setVideoModal] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchRecords = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await apiService.getRecords()
      // Map backend incident data to the format the UI expects
      const mapped = data.map((incident) => ({
        id: incident.id,
        examHall: 1,
        row: 0,
        col: 0,
        alertType: incident.labels || 'Detection',
        candidateId: incident.candidate_id || 'Unknown',
        confidence: 0,
        timestamp: new Date(incident.timestamp * 1000).toISOString(),
        videoPath: incident.clip_path || null,
        report: incident.report || '',
      }))
      setRecords(mapped)
    } catch (err) {
      console.error('Failed to fetch records:', err)
      setError('Failed to load records from backend')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchRecords()
    // Auto-refresh every 15 seconds
    const interval = setInterval(fetchRecords, 15000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">Video Records ({records.length})</h2>
        <button
          onClick={fetchRecords}
          className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition"
        >
          <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="bg-red-900 border border-red-700 text-red-100 px-4 py-3 rounded-lg">
          <p>{error}</p>
        </div>
      )}

      {loading && records.length === 0 ? (
        <div className="bg-gray-800 rounded-lg p-12 text-center border border-gray-700">
          <div className="text-gray-400 text-lg">Loading records...</div>
        </div>
      ) : records.length === 0 ? (
        <div className="bg-gray-800 rounded-lg p-12 text-center border border-gray-700">
          <div className="text-gray-400 text-lg">No records found</div>
          <p className="text-gray-500 text-sm mt-1">
            Recorded videos will appear here automatically when anomalies are detected
          </p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {records.map((record) => (
            <div
              key={record.id}
              className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700 hover:border-gray-500 transition"
            >
              {/* Thumbnail */}
              <div className="relative bg-black aspect-video flex items-center justify-center group cursor-pointer">
                <div className="absolute inset-0 bg-gradient-to-br from-gray-800 to-black flex items-center justify-center">
                  {record.videoPath ? (
                    <PlayCircle size={48} className="text-gray-500" />
                  ) : (
                    <div className="text-center">
                      <PlayCircle size={48} className="text-gray-600 mx-auto" />
                      <p className="text-gray-600 text-xs mt-2">No video</p>
                    </div>
                  )}
                </div>
                {record.videoPath && (
                  <button
                    onClick={() => setVideoModal(record)}
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition flex items-center justify-center bg-black bg-opacity-50"
                  >
                    <PlayCircle size={60} className="text-accent" />
                  </button>
                )}
              </div>

              {/* Info */}
              <div className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <p className="font-semibold text-white">
                      {record.alertType}
                    </p>
                    <p className="text-sm text-gray-400">Candidate: {record.candidateId}</p>
                  </div>
                </div>

                <div className="space-y-1 text-sm mb-3">
                  <p className="text-gray-400">
                    <span className="text-gray-500">Date:</span>{' '}
                    <span className="text-white">
                      {new Date(record.timestamp).toLocaleDateString()}
                    </span>
                  </p>
                  <p className="text-gray-400">
                    <span className="text-gray-500">Time:</span>{' '}
                    <span className="text-white">
                      {new Date(record.timestamp).toLocaleTimeString()}
                    </span>
                  </p>
                  {record.report && (
                    <p className="text-gray-400 text-xs mt-2 line-clamp-2">
                      <span className="text-gray-500">Report:</span>{' '}
                      <span className="text-gray-300">{record.report}</span>
                    </p>
                  )}
                </div>

                <button
                  onClick={() => record.videoPath && setVideoModal(record)}
                  disabled={!record.videoPath}
                  className={`w-full py-2 rounded transition font-semibold text-sm ${
                    record.videoPath
                      ? 'bg-primary hover:bg-blue-700 text-white'
                      : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  {record.videoPath ? 'Play Recording' : 'No Recording Available'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Video Modal */}
      {videoModal && (
        <div className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center p-4">
          <div className="bg-gray-800 rounded-lg max-w-4xl w-full overflow-hidden">
            {/* Header */}
            <div className="flex justify-between items-center p-4 border-b border-gray-700">
              <div>
                <h3 className="text-lg font-bold text-white">
                  {videoModal.alertType}
                </h3>
                <p className="text-sm text-gray-400">
                  Candidate: {videoModal.candidateId} • {new Date(videoModal.timestamp).toLocaleString()}
                </p>
              </div>
              <button
                onClick={() => setVideoModal(null)}
                className="p-2 hover:bg-gray-700 rounded transition"
              >
                <X size={24} className="text-white" />
              </button>
            </div>

            {/* Video Player (MJPEG stream) */}
            <div className="bg-black aspect-video flex items-center justify-center overflow-hidden">
              {videoModal.videoPath ? (
                <img
                  src={videoModal.videoPath}
                  alt="Recording playback"
                  className="w-full h-full object-contain"
                />
              ) : (
                <p className="text-gray-500">No video available for this record</p>
              )}
            </div>

            {/* Details */}
            <div className="p-4 bg-gray-750 grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-gray-500">Candidate</p>
                <p className="text-white font-semibold">{videoModal.candidateId}</p>
              </div>
              <div>
                <p className="text-gray-500">Date</p>
                <p className="text-white font-semibold">
                  {new Date(videoModal.timestamp).toLocaleDateString()}
                </p>
              </div>
              <div>
                <p className="text-gray-500">Time</p>
                <p className="text-white font-semibold">
                  {new Date(videoModal.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>

            {/* Report */}
            {videoModal.report && (
              <div className="p-4 border-t border-gray-700">
                <p className="text-gray-500 text-sm mb-1">VLM Report</p>
                <p className="text-gray-300 text-sm">{videoModal.report}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
