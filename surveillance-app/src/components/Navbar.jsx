import { useState, useEffect } from 'react'
import { Menu, X, LogOut, AlertCircle } from 'lucide-react'
import useAuthStore from '../context/authStore'
import useAlertStore from '../context/alertStore'

export default function Navbar({ onTabChange }) {
  const { user, logout } = useAuthStore()
  const { alerts } = useAlertStore()
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const unreadAlerts = alerts.filter((a) => !a.read).length

  const handleLogout = () => {
    logout()
    window.location.href = '/'
  }

  return (
    <nav className="bg-secondary border-b border-gray-700 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo */}
          <div className="flex items-center">
            <div className="text-xl font-bold text-accent flex items-center gap-2">
              <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
                <span className="text-secondary font-bold">A</span>
              </div>
              <span>AI Surveillance</span>
            </div>
          </div>

          {/* Desktop Menu */}
          {user && (
            <div className="hidden md:flex items-center space-x-1">
              <button
                onClick={() => onTabChange('feeds')}
                className="px-3 py-2 rounded-md text-sm font-medium text-gray-300 hover:bg-gray-700 hover:text-white transition"
              >
                Feeds
              </button>
              <button
                onClick={() => onTabChange('alerts')}
                className="px-3 py-2 rounded-md text-sm font-medium text-gray-300 hover:bg-gray-700 hover:text-white transition relative"
              >
                Alerts
                {unreadAlerts > 0 && (
                  <span className="absolute top-1 right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                    {unreadAlerts}
                  </span>
                )}
              </button>
              <button
                onClick={() => onTabChange('records')}
                className="px-3 py-2 rounded-md text-sm font-medium text-gray-300 hover:bg-gray-700 hover:text-white transition"
              >
                Records
              </button>
            </div>
          )}

          {/* Right side */}
          <div className="flex items-center space-x-4">
            {user ? (
              <>
                <div className="hidden md:flex items-center space-x-2">
                  <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                    <span className="text-white text-sm font-bold">
                      {user.username?.[0].toUpperCase()}
                    </span>
                  </div>
                  <span className="text-gray-300 text-sm">{user.username}</span>
                  {user.isAdmin && (
                    <span className="bg-red-500 px-2 py-1 rounded text-xs font-bold text-white">
                      ADMIN
                    </span>
                  )}
                </div>
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-2 px-3 py-2 rounded-md bg-red-600 hover:bg-red-700 text-white transition"
                >
                  <LogOut size={16} />
                  <span className="hidden sm:inline">Logout</span>
                </button>
              </>
            ) : null}

            {/* Mobile menu button */}
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="md:hidden inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-white hover:bg-gray-700"
            >
              {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isMenuOpen && user && (
        <div className="md:hidden border-t border-gray-700 bg-gray-800">
          <div className="px-2 pt-2 pb-3 space-y-1">
            <button
              onClick={() => {
                onTabChange('feeds')
                setIsMenuOpen(false)
              }}
              className="w-full text-left px-3 py-2 rounded-md text-gray-300 hover:bg-gray-700"
            >
              Feeds
            </button>
            <button
              onClick={() => {
                onTabChange('alerts')
                setIsMenuOpen(false)
              }}
              className="w-full text-left px-3 py-2 rounded-md text-gray-300 hover:bg-gray-700 flex justify-between items-center"
            >
              Alerts
              {unreadAlerts > 0 && (
                <span className="bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                  {unreadAlerts}
                </span>
              )}
            </button>
            <button
              onClick={() => {
                onTabChange('records')
                setIsMenuOpen(false)
              }}
              className="w-full text-left px-3 py-2 rounded-md text-gray-300 hover:bg-gray-700"
            >
              Records
            </button>
          </div>
        </div>
      )}
    </nav>
  )
}
