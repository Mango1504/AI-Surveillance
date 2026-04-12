import { useState } from 'react'
import { PlayCircle, X } from 'lucide-react'

// Mock records data - replace with actual API calls
const MOCK_RECORDS = [
  {
    id: 1,
    examHall: 1,
    row: 2,
    col: 3,
    alertType: 'Phone Detection',
    confidence: 0.95,
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    videoPath: '/videos/phone_001.avi',
  },
  {
    id: 2,
    examHall: 1,
    row: 1,
    col: 2,
    alertType: 'Phone Detection',
    confidence: 0.87,
    timestamp: new Date(Date.now() - 7200000).toISOString(),
    videoPath: '/videos/phone_002.avi',
  },
  {
    id: 3,
    examHall: 2,
    row: 3,
    col: 4,
    alertType: 'Phone Detection',
    confidence: 0.92,
    timestamp: new Date(Date.now() - 10800000).toISOString(),
    videoPath: '/videos/phone_003.avi',
  },
]

export default function RecordsView() {
  const [records] = useState(MOCK_RECORDS)
  const [videoModal, setVideoModal] = useState(null)
  const [filteredRecords, setFilteredRecords] = useState(MOCK_RECORDS)
  const [examHallFilter, setExamHallFilter] = useState('all')

  const handleExamHallFilter = (hall) => {
    setExamHallFilter(hall)
    if (hall === 'all') {
      setFilteredRecords(records)
    } else {
      setFilteredRecords(records.filter((r) => r.examHall === parseInt(hall)))
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">Video Records</h2>
        <div className="flex gap-2">
          <button
            onClick={() => handleExamHallFilter('all')}
            className={`px-4 py-2 rounded-lg transition ${
              examHallFilter === 'all'
                ? 'bg-primary text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            All
          </button>
          <button
            onClick={() => handleExamHallFilter('1')}
            className={`px-4 py-2 rounded-lg transition ${
              examHallFilter === '1'
                ? 'bg-primary text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Hall 1
          </button>
          <button
            onClick={() => handleExamHallFilter('2')}
            className={`px-4 py-2 rounded-lg transition ${
              examHallFilter === '2'
                ? 'bg-primary text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Hall 2
          </button>
        </div>
      </div>

      {filteredRecords.length === 0 ? (
        <div className="bg-gray-800 rounded-lg p-12 text-center border border-gray-700">
          <div className="text-gray-400 text-lg">No records found</div>
          <p className="text-gray-500 text-sm mt-1">
            Recorded videos will appear here
          </p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredRecords.map((record) => (
            <div
              key={record.id}
              className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700 hover:border-gray-500 transition"
            >
              {/* Thumbnail */}
              <div className="relative bg-black aspect-video flex items-center justify-center group cursor-pointer">
                <div className="absolute inset-0 bg-gradient-to-br from-gray-800 to-black flex items-center justify-center">
                  <PlayCircle size={48} className="text-gray-500" />
                </div>
                <button
                  onClick={() => setVideoModal(record)}
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition flex items-center justify-center bg-black bg-opacity-50"
                >
                  <PlayCircle size={60} className="text-accent" />
                </button>
              </div>

              {/* Info */}
              <div className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <p className="font-semibold text-white">
                      {record.alertType}
                    </p>
                    <p className="text-sm text-gray-400">ExamHall {record.examHall}</p>
                  </div>
                  <span className="bg-accent text-secondary px-2 py-1 rounded text-xs font-bold">
                    {(record.confidence * 100).toFixed(0)}%
                  </span>
                </div>

                <div className="space-y-1 text-sm mb-3">
                  <p className="text-gray-400">
                    <span className="text-gray-500">Position:</span>{' '}
                    <span className="text-white font-mono">
                      Row {record.row}, Col {record.col}
                    </span>
                  </p>
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
                </div>

                <button
                  onClick={() => setVideoModal(record)}
                  className="w-full py-2 bg-primary hover:bg-blue-700 text-white rounded transition font-semibold text-sm"
                >
                  Play Recording
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
                  Row {videoModal.row}, Col {videoModal.col} • ExamHall {videoModal.examHall}
                </p>
              </div>
              <button
                onClick={() => setVideoModal(null)}
                className="p-2 hover:bg-gray-700 rounded transition"
              >
                <X size={24} className="text-white" />
              </button>
            </div>

            {/* Video Player */}
            <div className="bg-black aspect-video flex items-center justify-center">
              <video
                src={videoModal.videoPath}
                controls
                autoPlay
                className="w-full h-full"
              />
            </div>

            {/* Details */}
            <div className="p-4 bg-gray-750 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-gray-500">Confidence</p>
                <p className="text-white font-semibold">
                  {(videoModal.confidence * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-gray-500">Position</p>
                <p className="text-white font-semibold">
                  R{videoModal.row}C{videoModal.col}
                </p>
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
          </div>
        </div>
      )}
    </div>
  )
}
