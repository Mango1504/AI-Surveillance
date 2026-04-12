import { useEffect, useState } from 'react'
import { apiService } from '../services/api'
import { Search, User, IdCard, Mail, Phone, Calendar, BookOpen, AlertCircle } from 'lucide-react'

export default function PersonIdentification() {
  const [applicants, setApplicants] = useState([])
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedPerson, setSelectedPerson] = useState(null)
  const [loading, setLoading] = useState(true)
  const [searchLoading, setSearchLoading] = useState(false)

  useEffect(() => {
    const fetchApplicants = async () => {
      setLoading(true)
      const data = await apiService.getApplicantsInfo()
      setApplicants(data.applicants || [])
      setLoading(false)
    }

    fetchApplicants()
  }, [])

  // Filter applicants based on search term
  const filteredApplicants = applicants.filter((person) => {
    const rollNumber = person.roll_number || ''
    const name = person.info?.name || ''
    const term = searchTerm.toLowerCase()
    return (
      rollNumber.toLowerCase().includes(term) ||
      name.toLowerCase().includes(term)
    )
  })

  const handleSearch = (e) => {
    setSearchLoading(true)
    setSearchTerm(e.target.value)
    setTimeout(() => setSearchLoading(false), 200)
  }

  const handleSelectPerson = (person) => {
    setSelectedPerson(person)
  }

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">👤 Person Identification</h1>
          <p className="text-gray-400">Search and identify students from the database</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Search Panel */}
          <div className="lg:col-span-1">
            <div className="bg-gradient-to-br from-purple-900 to-purple-800 border border-purple-600 rounded-lg p-6 sticky top-6">
              <div className="flex items-center gap-2 mb-4">
                <Search className="w-5 h-5 text-purple-300" />
                <h2 className="text-white font-bold">Search Database</h2>
              </div>

              {/* Search Input */}
              <div className="relative mb-6">
                <input
                  type="text"
                  placeholder="Roll number or name..."
                  value={searchTerm}
                  onChange={handleSearch}
                  className="w-full bg-purple-950 border border-purple-500 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-400"
                />
                {searchLoading && (
                  <div className="absolute right-3 top-3">
                    <div className="animate-spin">
                      <Search className="w-5 h-5 text-purple-400" />
                    </div>
                  </div>
                )}
              </div>

              {/* Results Count */}
              <div className="mb-4 text-sm text-purple-300">
                <p>Found: <span className="font-bold">{filteredApplicants.length}</span> of <span className="font-bold">{applicants.length}</span></p>
              </div>

              {/* Results List */}
              {loading ? (
                <div className="text-center py-8">
                  <p className="text-purple-300">Loading database...</p>
                </div>
              ) : filteredApplicants.length === 0 && searchTerm ? (
                <div className="text-center py-8">
                  <AlertCircle className="w-8 h-8 text-yellow-400 mx-auto mb-2" />
                  <p className="text-gray-400 text-sm">No results found</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {filteredApplicants.map((person) => (
                    <button
                      key={person.roll_number}
                      onClick={() => handleSelectPerson(person)}
                      className={`w-full text-left p-3 rounded-lg transition ${
                        selectedPerson?.roll_number === person.roll_number
                          ? 'bg-purple-600 border border-purple-400 text-white'
                          : 'bg-purple-950 border border-purple-600 text-gray-300 hover:bg-purple-900'
                      }`}
                    >
                      <p className="font-semibold text-sm">{person.info?.name}</p>
                      <p className="text-xs text-gray-400">{person.roll_number}</p>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Details Panel */}
          <div className="lg:col-span-2">
            {selectedPerson ? (
              <div className="bg-gradient-to-br from-blue-900 to-blue-800 border border-blue-600 rounded-lg overflow-hidden">
                {/* Header */}
                <div className="bg-gradient-to-r from-blue-700 to-blue-600 p-6 border-b border-blue-500">
                  <div className="flex items-center gap-4">
                    <div className="bg-blue-500 rounded-full p-4">
                      <User className="w-8 h-8 text-white" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-white">
                        {selectedPerson.info?.name}
                      </h3>
                      <p className="text-blue-200">
                        ID: {selectedPerson.roll_number}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Content */}
                <div className="p-6 space-y-6">
                  {/* Personal Info */}
                  <div>
                    <h4 className="text-white font-bold mb-4 flex items-center gap-2">
                      <IdCard className="w-5 h-5 text-blue-300" />
                      Personal Information
                    </h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-blue-950 p-4 rounded-lg">
                        <p className="text-gray-400 text-sm">Roll Number</p>
                        <p className="text-white font-semibold">{selectedPerson.roll_number}</p>
                      </div>
                      <div className="bg-blue-950 p-4 rounded-lg">
                        <p className="text-gray-400 text-sm">Status</p>
                        <p className="text-green-300 font-semibold">{selectedPerson.info?.status || 'Active'}</p>
                      </div>
                      {selectedPerson.info?.age && (
                        <div className="bg-blue-950 p-4 rounded-lg">
                          <p className="text-gray-400 text-sm">Age</p>
                          <p className="text-white font-semibold">{selectedPerson.info.age} years</p>
                        </div>
                      )}
                      {selectedPerson.info?.enrollment_year && (
                        <div className="bg-blue-950 p-4 rounded-lg">
                          <p className="text-gray-400 text-sm">Enrollment Year</p>
                          <p className="text-white font-semibold">{selectedPerson.info.enrollment_year}</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Academic Info */}
                  <div>
                    <h4 className="text-white font-bold mb-4 flex items-center gap-2">
                      <BookOpen className="w-5 h-5 text-blue-300" />
                      Academic Information
                    </h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-blue-950 p-4 rounded-lg">
                        <p className="text-gray-400 text-sm">Exam Hall</p>
                        <p className="text-white font-bold text-2xl">{selectedPerson.info?.exam_hall}</p>
                      </div>
                      <div className="bg-blue-950 p-4 rounded-lg">
                        <p className="text-gray-400 text-sm">Subject</p>
                        <p className="text-white font-semibold">{selectedPerson.info?.subject}</p>
                      </div>
                      {selectedPerson.info?.department && (
                        <div className="bg-blue-950 p-4 rounded-lg col-span-2">
                          <p className="text-gray-400 text-sm">Department</p>
                          <p className="text-white font-semibold">{selectedPerson.info.department}</p>
                        </div>
                      )}
                      {selectedPerson.info?.exam_date && (
                        <div className="bg-blue-950 p-4 rounded-lg col-span-2 flex items-center gap-2">
                          <Calendar className="w-5 h-5 text-blue-300" />
                          <div>
                            <p className="text-gray-400 text-sm">Exam Date</p>
                            <p className="text-white font-semibold">
                              {new Date(selectedPerson.info.exam_date).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Contact Info */}
                  {(selectedPerson.info?.email || selectedPerson.info?.phone) && (
                    <div>
                      <h4 className="text-white font-bold mb-4">Contact Information</h4>
                      <div className="space-y-3">
                        {selectedPerson.info?.email && (
                          <div className="bg-blue-950 p-4 rounded-lg flex items-center gap-3">
                            <Mail className="w-5 h-5 text-blue-300" />
                            <div>
                              <p className="text-gray-400 text-sm">Email</p>
                              <p className="text-white">{selectedPerson.info.email}</p>
                            </div>
                          </div>
                        )}
                        {selectedPerson.info?.phone && (
                          <div className="bg-blue-950 p-4 rounded-lg flex items-center gap-3">
                            <Phone className="w-5 h-5 text-blue-300" />
                            <div>
                              <p className="text-gray-400 text-sm">Phone</p>
                              <p className="text-white">{selectedPerson.info.phone}</p>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Status Badge */}
                  <div className="bg-gradient-to-r from-green-900 to-green-800 p-4 rounded-lg border border-green-600">
                    <p className="text-green-200 text-sm">✓ Student verified in database</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-lg p-12 text-center">
                <User className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                <p className="text-gray-400 text-lg mb-2">No person selected</p>
                <p className="text-gray-500">Search for a student in the left panel to view their details</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
