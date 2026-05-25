import { Shield, Video, Users, History, FileText, HelpCircle, Terminal } from 'lucide-react'

const navItems = [
  { id: 'live', label: 'Live Hub', icon: Video },
  { id: 'directory', label: 'Student Directory', icon: Users },
  { id: 'archive', label: 'Incident Archive', icon: History },
]

export default function Sidebar({ currentPage, onPageChange }) {
  return (
    <nav className="hidden md:flex flex-col h-full w-64 bg-surface-container-low border-r border-outline-variant/20 py-6 shrink-0 z-40">
      {/* Brand */}
      <div className="px-6 mb-8 flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-primary-container/20 border border-primary/30 flex items-center justify-center shadow-[0_0_15px_rgba(173,198,255,0.1)]">
          <Shield size={20} className="text-primary" />
        </div>
        <div>
          <h1 className="text-base font-bold text-on-surface leading-tight tracking-wide">PROCTOR CORE</h1>
          <p className="text-[11px] font-mono text-on-surface-variant tracking-widest">Hall Monitor v2.4</p>
        </div>
      </div>

      {/* Nav Links */}
      <div className="flex-1 flex flex-col gap-1 px-2">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = currentPage === item.id
          return (
            <button
              key={item.id}
              onClick={() => onPageChange(item.id)}
              className={`flex items-center gap-3 px-4 py-3 mx-2 rounded-xl transition-all duration-200 text-sm ${
                isActive
                  ? 'bg-primary-container text-on-primary-container shadow-[0_0_15px_rgba(173,198,255,0.2)] font-semibold'
                  : 'text-on-surface-variant hover:bg-surface-container-highest hover:text-primary'
              }`}
            >
              <Icon size={20} />
              <span>{item.label}</span>
            </button>
          )
        })}
      </div>

      {/* Report Button */}
      <div className="px-4 mb-4">
        <button className="w-full bg-surface-container-highest border border-outline-variant/50 text-on-surface py-2.5 rounded-lg font-mono text-xs tracking-widest hover:bg-primary-container/20 hover:text-primary transition-colors flex items-center justify-center gap-2">
          <FileText size={16} />
          GENERATE REPORT
        </button>
      </div>

      {/* Footer Links */}
      <div className="flex flex-col gap-1 px-6 border-t border-outline-variant/20 pt-4">
        <a href="#" className="flex items-center gap-3 text-on-surface-variant px-2 py-2 rounded-lg hover:text-primary transition-colors text-xs">
          <HelpCircle size={16} />
          <span className="font-mono uppercase tracking-widest">Support</span>
        </a>
        <a href="#" className="flex items-center gap-3 text-on-surface-variant px-2 py-2 rounded-lg hover:text-primary transition-colors text-xs">
          <Terminal size={16} />
          <span className="font-mono uppercase tracking-widest">Logs</span>
        </a>
      </div>
    </nav>
  )
}
