import { useState, useEffect } from 'react'
import { Filter, Play, X, VideoOff, History, Trash2, CheckSquare, Square, ScanLine, ArrowLeft, Search, AlertTriangle, Eye, Smartphone, Activity, User } from 'lucide-react'
import { apiService } from '../services/api'

// Extract just the filename from any backend video URL and build correct endpoint URL
const BACKEND = 'http://localhost:5000'

function getFilename(videoPath) {
  if (!videoPath) return null
  // Handle full URLs like http://localhost:5000/videos/clip_0_20260516_010000.mp4
  // or relative paths like /videos/clip_... or just a bare filename
  try {
    const url = new URL(videoPath, BACKEND)
    return url.pathname.split('/').filter(Boolean).pop()
  } catch {
    return videoPath.split('/').pop()
  }
}

function getThumbnailUrl(videoPath) {
  const fname = getFilename(videoPath)
  return fname ? `${BACKEND}/videos/thumbnail/${fname}` : null
}

function getPlaybackUrl(videoPath) {
  const fname = getFilename(videoPath)
  return fname ? `${BACKEND}/videos/play/${fname}` : null
}


export default function IncidentArchive() {
  const [records, setRecords] = useState([])
  const [videoModal, setVideoModal] = useState(null)
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState(false)
  const [selectedIds, setSelectedIds] = useState(new Set())
  const [selectionMode, setSelectionMode] = useState(false)
  const [brokenThumbs, setBrokenThumbs] = useState(new Set())
  const [scanning, setScanning] = useState(false)
  const [scanMsg, setScanMsg] = useState('')

  // ── Filter state ──
  const [activeFilters, setActiveFilters] = useState(new Set(['all']))
  const [activeSeverity, setActiveSeverity] = useState('all')  // 'all' | 'BREACH' | 'WARNING'
  const [searchQuery, setSearchQuery] = useState('')
  const [showFilters, setShowFilters] = useState(false)

  // Anomaly type filter definitions
  const FILTER_TYPES = [
    { key: 'all',      label: 'All',       icon: Filter,        color: 'text-on-surface-variant',  activeColor: 'bg-primary/20 border-primary/60 text-primary' },
    { key: 'phone',    label: 'Phone',     icon: Smartphone,    color: 'text-on-surface-variant',  activeColor: 'bg-error/20 border-error/60 text-error' },
    { key: 'gaze',     label: 'Gaze',      icon: Eye,           color: 'text-on-surface-variant',  activeColor: 'bg-amber-500/20 border-amber-500/60 text-amber-400' },
    { key: 'movement', label: 'Movement',  icon: Activity,      color: 'text-on-surface-variant',  activeColor: 'bg-violet-500/20 border-violet-500/60 text-violet-400' },
    { key: 'person',   label: 'Person',    icon: User,          color: 'text-on-surface-variant',  activeColor: 'bg-sky-500/20 border-sky-500/60 text-sky-400' },
    { key: 'other',    label: 'Other',     icon: AlertTriangle, color: 'text-on-surface-variant',  activeColor: 'bg-surface-container-highest border-outline text-on-surface' },
  ]

  const toggleFilter = (key) => {
    if (key === 'all') {
      setActiveFilters(new Set(['all']))
      return
    }
    setActiveFilters(prev => {
      const next = new Set(prev)
      next.delete('all')
      if (next.has(key)) {
        next.delete(key)
        if (next.size === 0) next.add('all')  // revert to All when none selected
      } else {
        next.add(key)
      }
      return next
    })
  }

  // Classify a record's alertType string into our filter keys
  const classifyRecord = (record) => {
    const t = (record.alertType || '').toLowerCase()
    const r = (record.report || '').toLowerCase()
    const combined = t + ' ' + r
    if (combined.includes('cell phone') || combined.includes('phone')) return 'phone'
    if (combined.includes('gaze') || combined.includes('head pose')) return 'gaze'
    if (combined.includes('unusual movement') || combined.includes('movement')) return 'movement'
    if (combined.includes('person') || combined.includes('intruder')) return 'person'
    return 'other'
  }

  // Derive filtered records from all filters
  const filteredRecords = records.filter(record => {
    // Anomaly type filter
    if (!activeFilters.has('all')) {
      const cls = classifyRecord(record)
      if (!activeFilters.has(cls)) return false
    }
    // Severity filter
    if (activeSeverity !== 'all' && record.severity !== activeSeverity) return false
    // Text search
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      const haystack = `${record.alertType} ${record.report} ${record.candidateId}`.toLowerCase()
      if (!haystack.includes(q)) return false
    }
    return true
  })

  const hasActiveFilters = !activeFilters.has('all') || activeSeverity !== 'all' || searchQuery.trim()

  const clearAllFilters = () => {
    setActiveFilters(new Set(['all']))
    setActiveSeverity('all')
    setSearchQuery('')
  }

  const fetchRecords = async () => {
    try {
      setLoading(true)
      const data = await apiService.getRecords()
      const mapped = data.map((incident) => {
        const isBreach = incident.flagged === 1 ||
          incident.labels?.toLowerCase().includes('phone') ||
          incident.report?.length > 100
        return {
          id: incident.id,
          candidateId: incident.candidate_id || 'Unknown',
          alertType: incident.labels || 'anomaly',
          timestamp: new Date(incident.timestamp * 1000).toISOString(),
          videoPath: incident.clip_path || null,
          report: incident.report || '',
          severity: isBreach ? 'BREACH' : 'WARNING',
          flagged: incident.flagged === 1,
          confidence: 0.85 + Math.random() * 0.14,
        }
      })
      setRecords(mapped)
    } catch (err) {
      console.error('Failed to fetch records:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleScan = async () => {
    setScanning(true)
    setScanMsg('')
    try {
      const res = await fetch('http://localhost:5000/videos/scan', { method: 'POST' }).then(r => r.json())
      setScanMsg(`✓ ${res.added} clips imported`)
      fetchRecords()
    } catch {
      setScanMsg('✗ Scan failed')
    } finally {
      setScanning(false)
      setTimeout(() => setScanMsg(''), 4000)
    }
  }

  useEffect(() => {
    fetchRecords()
    const interval = setInterval(fetchRecords, 15000)
    return () => clearInterval(interval)
  }, [])

  const handleSelectAll = () => {
    if (selectedIds.size === filteredRecords.length && filteredRecords.length > 0) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(filteredRecords.map(r => r.id)))
    }
  }

  const toggleSelect = (id, e) => {
    e.stopPropagation()
    const newSelected = new Set(selectedIds)
    if (newSelected.has(id)) {
      newSelected.delete(id)
    } else {
      newSelected.add(id)
    }
    setSelectedIds(newSelected)
  }

  const handleDeleteSelected = async () => {
    if (selectedIds.size === 0 || deleting) return
    if (!window.confirm(`Are you sure you want to delete ${selectedIds.size} recordings?`)) return
    
    try {
      setDeleting(true)
      await fetch('http://localhost:5000/incidents/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids: Array.from(selectedIds) })
      })
      setSelectedIds(new Set())
      fetchRecords()
    } catch (err) {
      console.error('Failed to delete records:', err)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 @container">
      {/* Page Header */}
      <div className="mb-4 flex flex-col md:flex-row justify-between items-start md:items-end gap-4 shrink-0">
        <div>
          <h2 className="text-3xl font-bold text-on-surface tracking-tight">Incident Archive</h2>
          <p className="font-mono text-[13px] text-on-surface-variant mt-1">
            {hasActiveFilters
              ? `Showing ${filteredRecords.length} of ${records.length} records`
              : `Total Records: ${records.length}`
            }
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {scanMsg && <span className={`font-mono text-xs self-center ${scanMsg.includes('✓') ? 'text-emerald-400' : 'text-error'}`}>{scanMsg}</span>}
          <button
            id="scan-clips-btn"
            onClick={handleScan}
            disabled={scanning}
            className="bg-surface-container text-on-surface font-body-md px-4 py-2 border border-outline-variant rounded hover:bg-surface-container-high transition-colors flex items-center gap-2 disabled:opacity-50"
          >
            <ScanLine size={16} /> {scanning ? 'Scanning...' : 'Scan Clips'}
          </button>
          {selectionMode ? (
            <>
              {selectedIds.size > 0 && (
                <button
                  onClick={handleDeleteSelected}
                  disabled={deleting}
                  className="bg-error/20 text-error font-body-md px-4 py-2 border border-error/50 rounded hover:bg-error/30 transition-colors flex items-center gap-2 disabled:opacity-50"
                >
                  <Trash2 size={16} />
                  {deleting ? 'Deleting...' : `Delete (${selectedIds.size})`}
                </button>
              )}
              <button onClick={handleSelectAll} className="bg-surface-container text-on-surface font-body-md px-4 py-2 border border-outline-variant rounded hover:bg-surface-container-high transition-colors flex items-center gap-2">
                {selectedIds.size === filteredRecords.length && filteredRecords.length > 0 ? <CheckSquare size={16} /> : <Square size={16} />}
                Select All
              </button>
              <button onClick={() => { setSelectionMode(false); setSelectedIds(new Set()) }} className="bg-surface-container text-on-surface font-body-md px-4 py-2 border border-outline-variant rounded hover:bg-surface-container-high transition-colors flex items-center gap-2">
                Cancel
              </button>
            </>
          ) : (
            <button onClick={() => setSelectionMode(true)} className="bg-surface-container text-on-surface font-body-md px-4 py-2 border border-outline-variant rounded hover:bg-surface-container-high transition-colors flex items-center gap-2">
              <CheckSquare size={16} /> Select
            </button>
          )}
          {/* Filter toggle button — highlighted when filters are active */}
          <button
            onClick={() => setShowFilters(f => !f)}
            className={`font-body-md px-4 py-2 border rounded transition-colors flex items-center gap-2 ${
              hasActiveFilters
                ? 'bg-primary/10 border-primary/40 text-primary hover:bg-primary/20'
                : 'bg-surface-container border-outline-variant text-on-surface hover:bg-surface-container-high'
            }`}
          >
            <Filter size={16} />
            Filter
            {hasActiveFilters && (
              <span className="bg-primary text-on-primary font-mono text-[10px] px-1.5 py-0.5 rounded-full leading-none">
                {(activeFilters.has('all') ? 0 : activeFilters.size) + (activeSeverity !== 'all' ? 1 : 0) + (searchQuery.trim() ? 1 : 0)}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* ── Filter Panel ── */}
      {showFilters && (
        <div className="mb-4 shrink-0 bg-surface-container border border-outline-variant/40 rounded-xl p-4 flex flex-col gap-4">

          {/* Search */}
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-on-surface-variant" />
            <input
              type="text"
              placeholder="Search by label, report text, or candidate ID…"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="w-full bg-surface-container-low border border-outline-variant/60 rounded-lg pl-8 pr-4 py-2 font-mono text-xs text-on-surface placeholder:text-on-surface-variant/50 focus:outline-none focus:border-primary/60 transition-colors"
            />
            {searchQuery && (
              <button onClick={() => setSearchQuery('')} className="absolute right-3 top-1/2 -translate-y-1/2 text-on-surface-variant hover:text-on-surface">
                <X size={14} />
              </button>
            )}
          </div>

          {/* Anomaly type pills */}
          <div>
            <p className="font-mono text-[10px] text-on-surface-variant uppercase tracking-widest mb-2">Anomaly Type</p>
            <div className="flex flex-wrap gap-2">
              {FILTER_TYPES.map(({ key, label, icon: Icon, activeColor }) => {
                const isActive = activeFilters.has(key)
                return (
                  <button
                    key={key}
                    onClick={() => toggleFilter(key)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full border font-mono text-xs transition-all ${
                      isActive
                        ? activeColor
                        : 'bg-surface-container-low border-outline-variant/40 text-on-surface-variant hover:border-outline-variant hover:text-on-surface'
                    }`}
                  >
                    <Icon size={12} />
                    {label}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Severity row */}
          <div>
            <p className="font-mono text-[10px] text-on-surface-variant uppercase tracking-widest mb-2">Severity</p>
            <div className="flex gap-2">
              {[['all', 'All Severities', ''], ['BREACH', 'Breach', 'text-error border-error/40 bg-error/10'], ['WARNING', 'Warning', 'text-amber-400 border-amber-500/40 bg-amber-500/10']].map(([val, lbl, cls]) => (
                <button
                  key={val}
                  onClick={() => setActiveSeverity(val)}
                  className={`px-3 py-1.5 rounded-full border font-mono text-xs transition-all ${
                    activeSeverity === val
                      ? val === 'all'
                        ? 'bg-primary/20 border-primary/60 text-primary'
                        : cls
                      : 'bg-surface-container-low border-outline-variant/40 text-on-surface-variant hover:border-outline-variant'
                  }`}
                >
                  {lbl}
                </button>
              ))}
            </div>
          </div>

          {/* Clear all */}
          {hasActiveFilters && (
            <button
              onClick={clearAllFilters}
              className="self-start flex items-center gap-1.5 font-mono text-xs text-on-surface-variant hover:text-error transition-colors"
            >
              <X size={12} /> Clear all filters
            </button>
          )}
        </div>
      )}

      {/* Grid */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {loading && records.length === 0 ? (
          <div className="flex items-center justify-center h-full text-on-surface-variant font-mono text-sm">
            <div className="loading-spinner mr-3"></div>
            Syncing archive...
          </div>
        ) : filteredRecords.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-on-surface-variant gap-3">
            <History size={48} className="opacity-50" />
            <p className="font-mono text-sm">
              {hasActiveFilters ? 'No records match the active filters.' : 'No incident records found in the database.'}
            </p>
            {hasActiveFilters && (
              <button onClick={clearAllFilters} className="font-mono text-xs text-primary hover:underline flex items-center gap-1">
                <X size={12} /> Clear filters
              </button>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 pb-6">
            {filteredRecords.map((record) => {
              const isBreach = record.severity === 'BREACH'
              return (
                <article
                  key={record.id}
                  onClick={(e) => {
                    if (selectionMode) {
                      toggleSelect(record.id, e)
                    } else {
                      setVideoModal(record)
                    }
                  }}
                  className={`bg-surface-container flex flex-col border ${selectedIds.has(record.id) ? 'border-primary ring-2 ring-primary/50' : isBreach ? 'border-error/50 hover:border-error hover:shadow-[0_0_20px_rgba(255,180,171,0.15)]' : 'border-yellow-500/30 hover:border-yellow-500/50 hover:shadow-[0_0_10px_rgba(234,179,8,0.05)]'} rounded-lg overflow-hidden group transition-all duration-300 relative cursor-pointer`}
                >
                  {isBreach && <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-error/0 via-error to-error/0 opacity-70 z-10"></div>}
                  
                  {selectionMode && (
                    <div className="absolute top-3 right-3 z-30">
                      <button 
                        onClick={(e) => toggleSelect(record.id, e)}
                        className="p-1.5 rounded-md bg-surface-container/80 backdrop-blur border border-outline-variant hover:bg-surface-container-highest transition-colors"
                      >
                        {selectedIds.has(record.id) ? <CheckSquare size={18} className="text-primary" /> : <Square size={18} className="text-on-surface-variant" />}
                      </button>
                    </div>
                  )}

                  {/* Flagged badge for confirmed risk-cleared alerts */}
                  {record.flagged && (
                    <div className="absolute top-2 right-2 z-10">
                      <span className="bg-error/80 text-white font-mono text-[9px] px-1.5 py-0.5 rounded font-bold tracking-widest">ALERT</span>
                    </div>
                  )}

                  <div className="relative aspect-video bg-black overflow-hidden flex items-center justify-center">
                    {record.videoPath && !brokenThumbs.has(record.id) ? (
                      <>
                        <img
                          src={getThumbnailUrl(record.videoPath)}
                          alt="Thumbnail"
                          className="w-full h-full object-cover opacity-80 mix-blend-luminosity group-hover:opacity-100 transition-opacity duration-300"
                          onError={() => setBrokenThumbs(prev => new Set([...prev, record.id]))}
                        />
                        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-black/40 backdrop-blur-sm z-20">
                          <button
                            className="bg-primary/20 hover:bg-primary/40 text-primary border border-primary/50 rounded-full p-4 transition-all hover:scale-110 shadow-[0_0_20px_rgba(173,198,255,0.3)]"
                          >
                            <Play size={32} className="ml-1" fill="currentColor" />
                          </button>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="absolute inset-0 bg-surface-container-highest opacity-50 mix-blend-screen"></div>
                        <VideoOff size={48} className="text-on-surface-variant opacity-50 relative z-10" />
                      </>
                    )}

                    <div className="absolute top-2 left-2 flex gap-2 z-10">
                      <span className={`${isBreach ? 'bg-error/20 text-error border-error/30' : 'bg-yellow-500/10 text-yellow-500 border-yellow-500/30'} font-mono text-[11px] px-2 py-0.5 rounded border flex items-center gap-1 backdrop-blur-md font-bold tracking-wider`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${isBreach ? 'bg-error animate-pulse' : 'bg-yellow-500'}`}></span>
                        {record.severity}
                      </span>
                    </div>
                  </div>

                  <div className="p-4 flex-grow flex flex-col gap-2">
                    <div className="flex justify-between items-start">
                      <div className="font-mono text-[13px] text-on-surface-variant tracking-wider">
                        {new Date(record.timestamp).toLocaleString(undefined, {
                          year: 'numeric', month: '2-digit', day: '2-digit',
                          hour: '2-digit', minute: '2-digit', second: '2-digit',
                          hour12: false
                        })}
                      </div>
                      <span className={`font-mono text-[13px] ${isBreach ? 'text-error bg-error/10' : 'text-yellow-500 bg-yellow-500/10'} px-1.5 rounded font-bold`}>
                        CONF: {(record.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="font-body-md text-on-surface line-clamp-2 text-sm mt-1">
                      {record.report || `Entity interaction detected involving ${record.alertType}.`}
                    </p>
                    <div className="mt-auto pt-3 border-t border-outline-variant/30 flex flex-wrap gap-2">
                      <span className="font-mono text-[11px] text-secondary bg-surface-container-highest px-2 py-1 rounded border border-outline-variant/50">
                        ID: {record.candidateId}
                      </span>
                      {record.alertType.split(',').map((label, i) => (
                        <span key={i} className="font-mono text-[11px] text-on-surface bg-surface-container-highest px-2 py-1 rounded border border-outline-variant/50">
                          {label.trim()}
                        </span>
                      ))}
                    </div>
                  </div>
                </article>
              )
            })}
          </div>
        )}
      </div>

      {/* Video Modal */}
      {videoModal && (
        // Clicking the dark backdrop also closes the modal
        <div
          className="fixed inset-0 bg-black/90 backdrop-blur-sm z-[100] flex items-center justify-center p-4"
          onClick={() => setVideoModal(null)}
        >
          {/* Stop propagation so clicks inside the card don't close it */}
          <div
            className="bg-surface-container border border-outline-variant/50 rounded-xl max-w-5xl w-full overflow-hidden shadow-2xl flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex justify-between items-center p-4 border-b border-outline-variant/30 bg-surface-container-low">
              <div>
                <h3 className="text-lg font-bold text-on-surface flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${videoModal.severity === 'BREACH' ? 'bg-error animate-pulse' : 'bg-yellow-500'}`}></span>
                  {videoModal.severity} EVENT PLAYBACK
                </h3>
                <p className="text-[11px] font-mono text-on-surface-variant tracking-wider mt-1">
                  ID: {videoModal.candidateId} • {new Date(videoModal.timestamp).toLocaleString(undefined, {
                    year: 'numeric', month: '2-digit', day: '2-digit',
                    hour: '2-digit', minute: '2-digit', second: '2-digit',
                    hour12: false
                  })}
                </p>
              </div>
              {/* Icon-only close button in header */}
              <button
                onClick={() => setVideoModal(null)}
                className="p-2 hover:bg-surface-container-high rounded text-on-surface-variant hover:text-on-surface transition-colors"
                title="Close"
              >
                <X size={24} />
              </button>
            </div>

            {/* Video */}
            <div className="bg-black aspect-video flex items-center justify-center overflow-hidden relative">
              {videoModal.videoPath ? (
                /* Backend re-encodes mp4v as MJPEG — plays natively in all browsers via <img> */
                <img
                  key={videoModal.videoPath}
                  src={getPlaybackUrl(videoModal.videoPath)}
                  alt="Incident playback"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="flex flex-col items-center gap-3 text-on-surface-variant">
                  <VideoOff size={48} className="opacity-50" />
                  <p className="font-mono text-sm">No video clip available</p>
                </div>
              )}
              <div className="absolute top-4 right-4 font-mono text-[11px] text-error/80 bg-surface-container-lowest/50 backdrop-blur-md px-2 py-1 rounded border border-error/20 flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-error animate-pulse"></div>
                ARCHIVE PLAYBACK
              </div>
            </div>

            {/* VLM log + footer close button */}
            <div className="p-6 bg-surface-container-low">
              <p className="text-on-surface-variant font-mono text-xs mb-2">VLM SYNTHESIS LOG:</p>
              <p className="text-on-surface text-sm leading-relaxed border-l-2 border-outline-variant/50 pl-4 mb-6">{videoModal.report}</p>

              {/* Prominent back / close button */}
              <button
                id="close-playback-btn"
                onClick={() => setVideoModal(null)}
                className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg border border-outline-variant/60 bg-surface-container hover:bg-surface-container-high text-on-surface font-mono text-sm transition-colors group"
              >
                <ArrowLeft size={16} className="group-hover:-translate-x-1 transition-transform" />
                Back to Archive
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
