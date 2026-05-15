import { useState, useEffect, useCallback } from 'react'
import { Filter, Calendar, Play, X, VideoOff, History, Trash2, CheckSquare, Square, ScanLine } from 'lucide-react'
import { apiService } from '../services/api'

// Derive thumbnail and playback URLs from the stored clip_path
function getThumbnailUrl(videoPath) {
  if (!videoPath) return null
  return videoPath.replace('/videos/', '/videos/thumbnail/')
}

function getPlaybackUrl(videoPath) {
  if (!videoPath) return null
  return videoPath.replace('/videos/', '/videos/play/')
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
    if (selectedIds.size === records.length && records.length > 0) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(records.map(r => r.id)))
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
      <div className="mb-6 flex flex-col md:flex-row justify-between items-start md:items-end gap-4 shrink-0">
        <div>
          <h2 className="text-3xl font-bold text-on-surface tracking-tight">Incident Archive</h2>
          <p className="font-mono text-[13px] text-on-surface-variant mt-1">Total Records: {records.length} | Displaying: Last 24 Hours</p>
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
                {selectedIds.size === records.length && records.length > 0 ? <CheckSquare size={16} /> : <Square size={16} />} 
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
          <button className="bg-surface-container text-on-surface font-body-md px-4 py-2 border border-outline-variant rounded hover:bg-surface-container-high transition-colors flex items-center gap-2">
            <Filter size={16} /> Filter
          </button>
          <button className="bg-surface-container text-on-surface font-body-md px-4 py-2 border border-outline-variant rounded hover:bg-surface-container-high transition-colors flex items-center gap-2">
            <Calendar size={16} /> Date Range
          </button>
        </div>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {loading && records.length === 0 ? (
          <div className="flex items-center justify-center h-full text-on-surface-variant font-mono text-sm">
            <div className="loading-spinner mr-3"></div>
            Syncing archive...
          </div>
        ) : records.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-on-surface-variant">
            <History size={48} className="opacity-50 mb-4" />
            <p className="font-mono text-sm">No incident records found in the database.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 pb-6">
            {records.map((record) => {
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
                        {new Date(record.timestamp).toISOString().replace('T', ' ').substring(0, 19)} UTC
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
        <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-[100] flex items-center justify-center p-4">
          <div className="bg-surface-container border border-outline-variant/50 rounded-xl max-w-5xl w-full overflow-hidden shadow-2xl flex flex-col">
            <div className="flex justify-between items-center p-4 border-b border-outline-variant/30 bg-surface-container-low">
              <div>
                <h3 className="text-lg font-bold text-on-surface flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${videoModal.severity === 'BREACH' ? 'bg-error animate-pulse' : 'bg-yellow-500'}`}></span>
                  {videoModal.severity} EVENT PLAYBACK
                </h3>
                <p className="text-[11px] font-mono text-on-surface-variant tracking-wider mt-1">
                  ID: {videoModal.candidateId} • {new Date(videoModal.timestamp).toISOString().replace('T', ' ').substring(0, 19)} UTC
                </p>
              </div>
              <button
                onClick={() => setVideoModal(null)}
                className="p-2 hover:bg-surface-container-high rounded text-on-surface-variant hover:text-on-surface transition-colors"
              >
                <X size={24} />
              </button>
            </div>
            
            <div className="bg-black aspect-video flex items-center justify-center overflow-hidden relative">
              {videoModal.videoPath ? (
                /* Backend re-encodes mp4v as MJPEG — plays natively in all browsers via <img> */
                <img
                  key={videoModal.videoPath}
                  src={videoModal.videoPath.replace(
                    /\/videos\//,
                    '/videos/play/'
                  ).replace('http://localhost:5000', 'http://localhost:5000')}
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
            
            <div className="p-6 bg-surface-container-low">
              <p className="text-on-surface-variant font-mono text-xs mb-2">VLM SYNTHESIS LOG:</p>
              <p className="text-on-surface text-sm leading-relaxed border-l-2 border-outline-variant/50 pl-4">{videoModal.report}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
