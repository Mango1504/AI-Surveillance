import { useEffect, useState } from 'react'
import { apiService } from '../services/api'
import { Eye, AlertTriangle } from 'lucide-react'

export default function FaceDetectionTab() {
  const [faceData, setFaceData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchFaces = async () => {
      setLoading(true)
      const data = await apiService.getAllFaces()
      setFaceData(data)
      setLoading(false)
    }

    fetchFaces()
    const interval = setInterval(fetchFaces, 2000) // Refresh every 2 seconds
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-400">Loading face data...</p>
      </div>
    )
  }

  if (!faceData) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-400">No data available</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Identified Faces */}
        <div className="bg-gradient-to-br from-blue-900 to-blue-800 border border-blue-600 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Identified Students</p>
              <p className="text-4xl font-bold text-blue-300">{faceData.total_identified}</p>
            </div>
            <Eye className="w-12 h-12 text-blue-500 opacity-50" />
          </div>
        </div>

        {/* Unknown Faces */}
        <div className={`border rounded-lg p-6 ${
          faceData.total_unknown > 0
            ? 'bg-gradient-to-br from-red-900 to-red-800 border-red-600'
            : 'bg-gradient-to-br from-green-900 to-green-800 border-green-600'
        }`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Unknown Faces</p>
              <p className={`text-4xl font-bold ${
                faceData.total_unknown > 0 ? 'text-red-300' : 'text-green-300'
              }`}>
                {faceData.total_unknown}
              </p>
            </div>
            <AlertTriangle className={`w-12 h-12 opacity-50 ${
              faceData.total_unknown > 0 ? 'text-red-500' : 'text-green-500'
            }`} />
          </div>
        </div>

        {/* Applicants Loaded */}
        <div className="bg-gradient-to-br from-purple-900 to-purple-800 border border-purple-600 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Database Loaded</p>
              <p className="text-4xl font-bold text-purple-300">{faceData.applicants_loaded}</p>
            </div>
            <div className="w-12 h-12 bg-purple-700 rounded-lg flex items-center justify-center">
              <span className="text-purple-300 font-bold text-xl">📁</span>
            </div>
          </div>
        </div>
      </div>

      {/* Identified Students List */}
      <div className="bg-blue-900 border border-blue-600 rounded-lg p-6">
        <h3 className="text-white font-bold text-lg mb-4 flex items-center gap-2">
          <Eye className="w-5 h-5 text-blue-300" />
          Identified Students Currently in Frame
        </h3>

        {faceData.identified_faces && faceData.identified_faces.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {faceData.identified_faces.map((face, idx) => (
              <div key={idx} className="bg-blue-950 border border-blue-700 p-4 rounded-lg">
                <div className="mb-3">
                  <p className="text-blue-100 font-semibold">{face.name}</p>
                  <p className="text-gray-400 text-sm">Roll: {face.roll_number}</p>
                </div>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-400">
                    Hall: <span className="text-blue-300 font-semibold">{face.exam_hall}</span>
                  </p>
                  <p className="text-gray-400">
                    Subject: <span className="text-blue-300 font-semibold">{face.subject}</span>
                  </p>
                  <p className="text-gray-400">
                    Position: <span className="text-green-400 font-semibold">R{face.grid_row}C{face.grid_col}</span>
                  </p>
                  <div className="pt-2 border-t border-blue-800">
                    <p className="text-gray-400">
                      Confidence: <span className="text-green-400 font-bold">{(face.confidence * 100).toFixed(0)}%</span>
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-400 text-center py-8">No students currently identified</p>
        )}
      </div>

      {/* Unknown Faces Alert */}
      {faceData.unknown_faces && faceData.unknown_faces.length > 0 && (
        <div className="bg-red-900 border border-red-600 rounded-lg p-6">
          <h3 className="text-white font-bold text-lg mb-4 flex items-center gap-2 text-red-200">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            🚨 Unknown Faces Detected - SECURITY ALERT
          </h3>

          <div className="space-y-3">
            {faceData.unknown_faces.map((face, idx) => (
              <div key={idx} className="bg-red-950 border border-red-500 p-4 rounded-lg">
                <div className="flex items-start justify-between mb-2">
                  <p className="text-red-200 font-semibold">{face.message}</p>
                  <span className="bg-red-600 text-white text-xs px-2 py-1 rounded">
                    {face.alert_level}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <p className="text-gray-400">
                    Confidence: <span className="text-red-300">{(face.confidence * 100).toFixed(0)}%</span>
                  </p>
                  <p className="text-gray-400">
                    Time: <span className="text-red-300">{new Date(face.timestamp).toLocaleTimeString()}</span>
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No threats message */}
      {faceData.total_unknown === 0 && (
        <div className="bg-gradient-to-r from-green-900 to-green-800 border border-green-600 rounded-lg p-6 text-center">
          <p className="text-green-200 font-semibold text-lg">✓ All Clear - No Unknown Faces Detected</p>
          <p className="text-green-300 text-sm mt-2">Security status is normal</p>
        </div>
      )}
    </div>
  )
}
