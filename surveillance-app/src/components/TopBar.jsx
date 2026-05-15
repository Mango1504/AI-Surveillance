import { Bell, LayoutGrid, Settings, AlertTriangle, LogOut } from 'lucide-react'
import useAuthStore from '../context/authStore'

export default function TopBar() {
  const { user, logout } = useAuthStore()

  const handleLogout = () => {
    logout()
    window.location.href = '/'
  }

  return (
    <header className="flex justify-between items-center w-full px-6 h-16 bg-surface-dim/80 backdrop-blur-xl border-b border-outline-variant/30 shadow-sm shadow-primary/10 shrink-0 z-30">
      <div className="flex items-center gap-4">
        <h1 className="text-xl font-bold text-primary tracking-tight uppercase hidden lg:block">
          AEGIS SURVEILLANCE
        </h1>
      </div>

      <div className="flex items-center gap-4">
        {/* System Status */}
        <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-container border border-outline-variant/50">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="font-mono text-[11px] text-tertiary uppercase tracking-widest">System: ONLINE</span>
        </div>

        {/* Panic Alert */}
        <button className="bg-error-container/20 text-error border border-error/50 px-4 py-1.5 rounded font-mono text-xs tracking-widest hover:bg-error-container hover:text-on-error-container transition-colors shadow-[0_0_10px_rgba(255,180,171,0.1)]">
          PANIC ALERT
        </button>

        <div className="h-6 w-px bg-outline-variant/50 mx-1" />

        {/* Icon Buttons */}
        <div className="flex items-center gap-1">
          <button className="w-9 h-9 flex items-center justify-center rounded-full text-on-surface-variant hover:bg-surface-container-high/50 hover:text-primary transition-all duration-200">
            <Bell size={18} />
          </button>
          <button className="w-9 h-9 flex items-center justify-center rounded-full text-on-surface-variant hover:bg-surface-container-high/50 hover:text-primary transition-all duration-200">
            <LayoutGrid size={18} />
          </button>
          <button className="w-9 h-9 flex items-center justify-center rounded-full text-on-surface-variant hover:bg-surface-container-high/50 hover:text-primary transition-all duration-200">
            <Settings size={18} />
          </button>
        </div>

        {/* User */}
        {user && (
          <div className="flex items-center gap-2 ml-2">
            <div className="w-8 h-8 rounded-full bg-primary-container/30 border border-primary/40 flex items-center justify-center">
              <span className="text-primary text-xs font-bold">{user.username?.[0]?.toUpperCase()}</span>
            </div>
            <button
              onClick={handleLogout}
              className="w-8 h-8 flex items-center justify-center rounded-full text-on-surface-variant hover:bg-error/20 hover:text-error transition-all"
              title="Logout"
            >
              <LogOut size={16} />
            </button>
          </div>
        )}
      </div>
    </header>
  )
}
