import { useEffect, useState } from 'react'
import { apiService } from '../services/api'
import { Users, CheckCircle, RefreshCw } from 'lucide-react'

export default function IdentifiedStudents() {
  const [students, setStudents] = useState([])
  const [loading, setLoading] = useState(true)
  const [totalApplicants, setTotalApplicants] = useState(0)

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      const [identified, applicants] = await Promise.all([
        apiService.getIdentifiedFaces(),
        apiService.getApplicantsInfo()
      ])
      setStudents(identified.identified_students || [])
      setTotalApplicants(applicants.total_applicants || 0)
      setLoading(false)
    }

    fetchData()
    const interval = setInterval(fetchData, 2000) // Refresh every 2 seconds
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="bg-gradient-to-br from-blue-900 to-blue-800 border border-blue-600 rounded-lg p-6 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Users className="w-8 h-8 text-blue-300" />
          <div>
            <h3 className="text-white font-bold text-lg">Identified Students</h3>
            <p className="text-blue-200 text-sm">
              Currently recognized in frame
            </p>
          </div>
        </div>
        <div className="bg-blue-700 text-white px-4 py-2 rounded-full font-bold text-lg">
          {students.length}/{totalApplicants}
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-8">
          <RefreshCw className="w-5 h-5 text-blue-300 animate-spin mr-2" />
          <p className="text-blue-200">Scanning...</p>
        </div>
      ) : students.length === 0 ? (
        <div className="text-blue-300 py-8 text-center">
          <p>No students currently identified</p>
        </div>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {students.map((student, idx) => (
            <div
              key={idx}
              className="bg-blue-950 bg-opacity-50 border border-blue-500 p-4 rounded-lg hover:bg-opacity-70 transition"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <div>
                    <p className="text-blue-100 font-semibold">{student.name}</p>
                    <p className="text-gray-400 text-sm">Roll: {student.roll_number}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-green-400 font-bold text-lg">
                    {(student.confidence * 100).toFixed(0)}%
                  </p>
                  <p className="text-gray-400 text-xs">Confidence</p>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 text-xs">
                <div>
                  <p className="text-gray-400">Hall</p>
                  <p className="text-blue-300 font-semibold">{student.exam_hall}</p>
                </div>
                <div>
                  <p className="text-gray-400">Subject</p>
                  <p className="text-blue-300 font-semibold">{student.subject}</p>
                </div>
                <div>
                  <p className="text-gray-400">Position</p>
                  <p className="text-blue-300 font-semibold">R{student.grid_row}C{student.grid_col}</p>
                </div>
              </div>

              <div className="mt-2 text-xs text-gray-500 border-t border-blue-700 pt-2">
                {new Date(student.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-4 pt-4 border-t border-blue-700 text-xs text-gray-400">
        <p>Total Registered: <span className="text-blue-300 font-bold">{totalApplicants}</span></p>
      </div>
    </div>
  )
}
