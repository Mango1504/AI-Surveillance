import { useNavigate } from 'react-router-dom'
import { Shield, LogOut, Settings, Activity, Database, Monitor } from 'lucide-react'
import useAuthStore from '../context/authStore'
import AdminSettings from '../components/AdminSettings'

export default function AdminPanel() {
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/admin-login')
  }

  return (
    <div className="bg-background text-on-background h-screen w-full flex overflow-hidden font-body-md">

      {/* Admin Sidebar */}
      <nav className="hidden md:flex flex-col h-full w-64 bg-surface-container-low border-r border-red-900/30 py-6 shrink-0 z-40">
        {/* Brand */}
        <div className="px-6 mb-8 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-red-900/30 border border-red-500/40 flex items-center justify-center shadow-[0_0_15px_rgba(255,80,80,0.15)]">
            <Shield size={20} className="text-red-400" />
          </div>
          <div>
            <h1 className="text-base font-bold text-on-surface leading-tight tracking-wide">ADMIN PANEL</h1>
            <p className="text-[11px] font-mono text-red-400/70 tracking-widest">Restricted Access</p>
          </div>
        </div>

        {/* Nav Items */}
        <div className="flex-1 flex flex-col gap-1 px-2">
          {[
            { label: 'System Config', icon: Settings, active: true },
            { label: 'Live Telemetry', icon: Activity, active: false },
            { label: 'Identity DB', icon: Database, active: false },
            { label: 'Hardware', icon: Monitor, active: false },
          ].map((item) => {
            const Icon = item.icon
            return (
              <div
                key={item.label}
                className={`flex items-center gap-3 px-4 py-3 mx-2 rounded-xl text-sm cursor-default transition-all duration-200 ${
                  item.active
                    ? 'bg-red-900/30 text-red-300 border border-red-800/40 font-semibold'
                    : 'text-on-surface-variant opacity-50'
                }`}
              >
                <Icon size={20} />
                <span>{item.label}</span>
              </div>
            )
          })}
        </div>

        {/* Admin User + Logout */}
        <div className="px-4 border-t border-red-900/30 pt-4">
          <div className="flex items-center gap-3 px-2 py-2 mb-2">
            <div className="w-8 h-8 rounded-full bg-red-900/40 border border-red-500/50 flex items-center justify-center">
              <span className="text-red-300 text-xs font-bold">{user?.username?.[0]?.toUpperCase()}</span>
            </div>
            <div>
              <p className="text-sm font-semibold text-on-surface">{user?.username}</p>
              <p className="text-[10px] font-mono text-red-400 uppercase tracking-widest">Administrator</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-2 px-4 py-2 rounded-lg bg-red-900/20 border border-red-800/40 text-red-400 hover:bg-red-900/40 transition-colors font-mono text-xs tracking-widest"
          >
            <LogOut size={15} />
            LOGOUT
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">

        {/* Admin Top Bar */}
        <header className="flex justify-between items-center w-full px-6 h-16 bg-surface-dim/80 backdrop-blur-xl border-b border-red-900/20 shadow-sm shrink-0 z-30">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="font-mono text-xs text-red-400 uppercase tracking-widest">Admin Session Active</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-container border border-red-900/30">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="font-mono text-[11px] text-tertiary uppercase tracking-widest">System: ONLINE</span>
            </div>
            {/* Mobile logout */}
            <button
              onClick={handleLogout}
              className="md:hidden flex items-center gap-2 px-3 py-1.5 rounded-lg bg-red-900/20 border border-red-800/40 text-red-400 hover:bg-red-900/40 transition-colors font-mono text-xs"
            >
              <LogOut size={14} />
              Logout
            </button>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6 bg-surface-dim relative z-10 flex flex-col">
          <AdminSettings />
        </main>
      </div>
    </div>
  )
}
