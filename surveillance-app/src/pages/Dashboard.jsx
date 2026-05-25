import { useState } from 'react'
import Sidebar from '../components/Sidebar'
import TopBar from '../components/TopBar'
import LiveHub from '../components/LiveHub'
import IncidentArchive from '../components/IncidentArchive'
import StudentDirectory from '../components/StudentDirectory'

export default function Dashboard() {
  const [currentPage, setCurrentPage] = useState('live')

  // Removed isAdmin check to allow local users to access the Admin panel

  return (
    <div className="bg-background text-on-background h-screen w-full flex overflow-hidden font-body-md">
      <Sidebar currentPage={currentPage} onPageChange={setCurrentPage} />
      
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">
        <TopBar />
        
        {/* Content Canvas */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6 bg-surface-dim relative z-10 flex flex-col">
          {currentPage === 'live' && <LiveHub />}
          {currentPage === 'directory' && <StudentDirectory />}
          {currentPage === 'archive' && <IncidentArchive />}
        </main>
      </div>
    </div>
  )
}
