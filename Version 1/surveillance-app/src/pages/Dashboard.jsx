import { useState } from 'react'
import Navbar from '../components/Navbar'
import FeedView from '../components/FeedView'
import AlertsTab from '../components/AlertsTab'
import RecordsView from '../components/RecordsView'

export default function Dashboard() {
  const [currentTab, setCurrentTab] = useState('feeds')
  const [selectedExamHall, setSelectedExamHall] = useState(1)

  const examHalls = [1, 2, 3, 4]

  return (
    <div className="min-h-screen bg-gray-900">
      <Navbar onTabChange={setCurrentTab} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Exam Hall Selector (for Feeds tab) */}
        {currentTab === 'feeds' && (
          <div className="mb-6">
            <h2 className="text-lg font-semibold text-white mb-3">Select Exam Hall</h2>
            <div className="flex gap-3 flex-wrap">
              {examHalls.map((hall) => (
                <button
                  key={hall}
                  onClick={() => setSelectedExamHall(hall)}
                  className={`px-4 py-2 rounded-lg font-semibold transition ${
                    selectedExamHall === hall
                      ? 'bg-primary text-white'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  Hall {hall}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="mb-6 border-b border-gray-700 pb-4">
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => setCurrentTab('feeds')}
              className={`px-6 py-3 rounded-lg font-semibold transition ${
                currentTab === 'feeds'
                  ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              📹 Live Feed
            </button>
            <button
              onClick={() => setCurrentTab('alerts')}
              className={`px-6 py-3 rounded-lg font-semibold transition ${
                currentTab === 'alerts'
                  ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              ⚠️ Alerts
            </button>
            <button
              onClick={() => setCurrentTab('records')}
              className={`px-6 py-3 rounded-lg font-semibold transition ${
                currentTab === 'records'
                  ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              📹 Records
            </button>
          </div>
        </div>

        {/* Tab Content */}
        <div className="animate-fadeIn">
          {currentTab === 'feeds' && <FeedView examHall={selectedExamHall} />}
          {currentTab === 'alerts' && <AlertsTab />}
          {currentTab === 'records' && <RecordsView />}
        </div>
      </div>
    </div>
  )
}
