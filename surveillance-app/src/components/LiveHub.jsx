import { useState, useEffect } from 'react'
import { Video, AlertTriangle, UserCheck } from 'lucide-react'
import { useDetectionStatus, useGridInfo } from '../hooks/useDetection'
import useAlertStore from '../context/alertStore'
import { apiService } from '../services/api'

export default function LiveHub() {
  const [streamError, setStreamError] = useState(false)
  const [examHall] = useState(1)
  const { status, error } = useDetectionStatus(examHall)
  const { gridInfo } = useGridInfo()
  const { alerts, addAlert } = useAlertStore()
  
  const [identifiedFaces, setIdentifiedFaces] = useState([])

  // Trigger alert when any backend anomaly is detected
  useEffect(() => {
    if ((status?.anomaly_detected || status?.phone_detected) && status.detections?.length > 0) {
      status.detections.forEach((det) => {
        if (det.grid_row && det.label) {
          addAlert({
            examHall,
            row: det.grid_row,
            col: det.grid_col,
            confidence: det.confidence,
            timestamp: status.timestamp,
            alert_type: (det.class || det.label).toUpperCase(),
            clip_path: status.clip_path,
          })
        }
      })
    }
  }, [status?.anomaly_detected, status?.phone_detected, status?.detections, status?.timestamp, status?.clip_path, examHall, addAlert])

  // Fetch identified students from real backend endpoint
  useEffect(() => {
    const fetchFaces = async () => {
      try {
        const data = await apiService.getIdentifiedStudents()
        setIdentifiedFaces(data.identified_students || [])
      } catch {
        // Silently ignore — endpoint may not have data
      }
    }
    fetchFaces()
    const interval = setInterval(fetchFaces, 5000)
    return () => clearInterval(interval)
  }, [])

  const renderFeedGrid = () => {
    if (!gridInfo) return null
    const { rows, cols } = gridInfo
    const cells = []
    for (let r = 1; r <= rows; r++) {
      for (let c = 1; c <= cols; c++) {
        const isAlert = status?.detections?.some(d => d.grid_row === r && d.grid_col === c)
        cells.push(
          <div key={`${r}-${c}`} className={`grid-cell relative ${isAlert ? 'border-error/40 bg-error/5 shadow-[0_0_15px_rgba(255,180,171,0.2)] z-20 pulse-alert' : ''}`}>
             <div className="absolute top-1 right-1 font-data-sm text-[10px] text-primary/40 bg-surface-container-lowest/80 px-1 rounded backdrop-blur-sm">R{r}C{c}</div>
          </div>
        )
      }
    }
    return cells
  }

  // Use only recent active threats (e.g. last 5)
  const activeThreats = alerts.slice(0, 5)

  return (
    <div className="flex-1 flex flex-col lg:flex-row gap-6 h-full overflow-hidden">
      {/* Central Video Feed */}
      <div className="flex-1 flex flex-col gap-1 h-full relative min-w-0">
        <div className="flex justify-between items-end mb-2 shrink-0">
          <div>
            <h1 className="font-headline-md text-xl text-on-surface flex items-center gap-2">
              <Video className="text-primary" size={24} />
              MAIN HALLWAY - SECTOR {examHall}
            </h1>
            <p className="font-mono text-[11px] text-on-surface-variant uppercase mt-1">
              FEED_ID: CH-0{examHall} | RES: 1080p | FPS: {status?.fps?.toFixed(1) ?? '–'} | LATENCY: {status?.alert_latency_ms ?? 0}ms
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={async () => {
                const newState = !status?.manual_record;
                try {
                  await fetch('http://localhost:5000/manual-record', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: newState })
                  });
                } catch (e) { console.error(e) }
              }}
              className={`font-mono text-xs px-3 py-1.5 rounded flex items-center gap-2 transition-colors ${
                status?.manual_record 
                ? 'bg-error text-on-error shadow-[0_0_15px_rgba(248,113,113,0.3)] border border-error' 
                : 'bg-surface-container border border-outline-variant text-on-surface-variant hover:bg-surface-container-high'
              }`}
            >
              <div className={`w-2 h-2 rounded-full ${status?.manual_record ? 'bg-white animate-pulse' : 'bg-error'}`}></div>
              {status?.manual_record ? 'STOP REC' : 'START REC'}
            </button>
            <button
              onClick={async () => {
                const newState = !(status?.auto_record !== false);
                try {
                  await fetch('http://localhost:5000/toggle-recording', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: newState })
                  });
                } catch (e) { console.error(e) }
              }}
              className={`font-mono text-xs px-3 py-1.5 rounded flex items-center gap-2 transition-colors ${
                status?.auto_record !== false 
                ? 'bg-error/20 border border-error/50 text-error shadow-[0_0_10px_rgba(248,113,113,0.1)]' 
                : 'bg-surface-container border border-outline-variant text-on-surface-variant'
              }`}
            >
              <div className={`w-2 h-2 rounded-full ${status?.auto_record !== false ? 'bg-error animate-pulse' : 'bg-outline'}`}></div>
              {status?.auto_record !== false ? 'AUTO-REC: ON' : 'AUTO-REC: OFF'}
            </button>
            <div className={`font-mono text-xs px-3 py-1.5 rounded flex items-center gap-2 ${
              error ? 'bg-error/20 border border-error/50 text-error' :
              status?.recording ? 'bg-error/20 border border-error/50 text-error shadow-[0_0_10px_rgba(248,113,113,0.15)]' :
              'bg-surface-container border border-emerald-500/30 text-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.1)]'
            }`}>
              <span className={`w-2 h-2 rounded-full animate-pulse ${
                error ? 'bg-error' : status?.recording ? 'bg-error' : 'bg-emerald-500'
              }`}></span>
              {error ? 'OFFLINE' : status?.recording ? 'REC ACTIVE' : status?.anomaly_detected ? 'ANOMALY' : 'SYSTEM CLEAR'}
            </div>
          </div>
        </div>
        {/* Risk Score Bar (Layer 4) */}
        {(() => {
          const score = status?.risk_score ?? 0
          const pct = Math.round(score * 100)
          const sustained = status?.risk_sustained_secs ?? 0
          const barColor = pct >= 65 ? '#f87171' : pct >= 40 ? '#fbbf24' : '#34d399'
          return (
            <div className="shrink-0 px-1 mb-2">
              <div className="flex justify-between items-center mb-1">
                <span className="font-mono text-[10px] text-on-surface-variant uppercase tracking-wider">Risk Score · L4</span>
                <span className="font-mono text-[10px]" style={{ color: barColor }}>
                  {pct}% {sustained > 0 ? `· ${sustained.toFixed(1)}s sustained` : ''}
                  {pct >= 65 ? ' · ALERT' : ''}
                </span>
              </div>
              <div className="w-full h-2 bg-surface-container-highest rounded-full overflow-hidden relative">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{ width: `${pct}%`, backgroundColor: barColor }}
                />
                {/* Threshold marker at 65% */}
                <div className="absolute top-0 h-full w-[1px] bg-white/40" style={{ left: '65%' }} />
              </div>
            </div>
          )
        })()}

        {/* Video Container */}
        <div className="flex-1 bg-surface-container-lowest border border-outline-variant/30 rounded-xl overflow-hidden relative shadow-lg min-h-0">
          <img
            src={`http://localhost:5000/stream`}
            alt="Live Feed"
            className="w-full h-full object-cover"
            onError={() => setStreamError(true)}
            onLoad={() => setStreamError(false)}
          />
          
          {streamError && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm z-30">
              <AlertTriangle className="text-error mb-2 animate-pulse" size={48} />
              <p className="text-on-surface font-headline-md">SIGNAL LOST</p>
              <p className="text-on-surface-variant font-mono text-sm">Attempting to reconnect...</p>
            </div>
          )}

          {/* Grid Overlay */}
          {gridInfo && (
            <div
              className="feed-grid-overlay"
              style={{
                gridTemplateRows: `repeat(${gridInfo.rows}, 1fr)`,
                gridTemplateColumns: `repeat(${gridInfo.cols}, 1fr)`,
              }}
            >
              {renderFeedGrid()}
            </div>
          )}

          {/* HUD Elements */}
          <div className="absolute top-4 left-4 font-mono text-xs text-primary/80 bg-surface-container-lowest/50 backdrop-blur-md px-2 py-1 rounded border border-primary/20 z-20 flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${status?.recording ? 'bg-error animate-pulse' : 'bg-primary'}`}></div>
            {status?.recording ? 'REC' : 'LIVE'} • {new Date().toLocaleTimeString()}
          </div>
          <div className="absolute bottom-4 right-4 font-mono text-xs text-on-surface-variant bg-surface-container-lowest/50 backdrop-blur-md px-2 py-1 rounded border border-outline-variant/20 z-20">
            ZOOM: 1.0x | PTZ: AUTO
          </div>
        </div>
      </div>

      {/* Side Panels */}
      <div className="w-full lg:w-80 flex flex-col gap-6 shrink-0 h-full overflow-hidden">
        {/* Active Threats Sidebar */}
        <div className="flex-1 bg-surface-container border border-outline-variant/30 rounded-xl flex flex-col overflow-hidden shadow-md min-h-0">
          <div className="p-3 border-b border-outline-variant/30 bg-surface-container-low flex items-center justify-between shrink-0">
            <h3 className="font-mono text-xs text-on-surface flex items-center gap-2 font-bold tracking-wider">
              <AlertTriangle className="text-error" size={16} />
              ACTIVE THREATS
            </h3>
            <span className="text-[10px] font-mono bg-surface-container-highest px-2 py-0.5 rounded text-on-surface-variant font-bold">
              {activeThreats.length} DETECTED
            </span>
          </div>
          <div className="flex-1 overflow-y-auto p-2 flex flex-col gap-2">
            {activeThreats.length === 0 ? (
              <div className="flex items-center justify-center h-full text-on-surface-variant font-mono text-xs">
                No active threats
              </div>
            ) : (
              activeThreats.map((alert, idx) => (
                <div key={alert.id || idx} className="bg-surface-container-low border-l-2 border-error p-3 rounded-r-lg hover:bg-surface-container-highest transition-colors cursor-pointer group">
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-mono text-xs font-bold text-error group-hover:text-error-container transition-colors truncate pr-2">
                      {alert.alert_type}
                    </span>
                    <span className="font-mono text-[10px] text-on-surface-variant bg-surface-container px-1.5 py-0.5 rounded border border-outline-variant/30 shrink-0">
                      R{alert.row}C{alert.col}
                    </span>
                  </div>
                  <div className="flex justify-between items-end mt-2">
                    <span className="text-[10px] font-mono text-on-surface-variant">CONF: {(alert.confidence * 100).toFixed(1)}%</span>
                    <span className="text-[10px] font-mono text-primary">{new Date(alert.timestamp).toLocaleTimeString()}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Identified in Frame Panel */}
        <div className="flex-1 bg-surface-container border border-outline-variant/30 rounded-xl flex flex-col overflow-hidden shadow-md min-h-0">
          <div className="p-3 border-b border-outline-variant/30 bg-surface-container-low flex items-center justify-between shrink-0">
            <h3 className="font-mono text-xs text-on-surface flex items-center gap-2 font-bold tracking-wider">
              <UserCheck className="text-primary" size={16} />
              IDENTITY MANAGER
            </h3>
          </div>
          <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
            {identifiedFaces.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-on-surface-variant font-mono text-xs text-center gap-2">
                <UserCheck size={24} className="opacity-30" />
                <span>No verified identities in frame</span>
                <span className="text-[10px] text-on-surface-variant/50">DeepFace biometrics will appear here when matched</span>
              </div>
            ) : (
              identifiedFaces.map((face, idx) => (
                <div key={idx} className="flex items-center gap-3 p-2 bg-surface-container-low rounded-lg border border-outline-variant/20 hover:border-primary/30 transition-colors">
                  <div className="w-10 h-10 rounded bg-surface-container-highest border border-outline-variant/30 flex items-center justify-center shrink-0">
                    <UserCheck className="text-outline" size={20} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-mono text-xs font-bold text-on-surface truncate">{face.name?.toUpperCase() || 'UNKNOWN'}</p>
                    <p className="text-[10px] text-on-surface-variant font-mono mt-0.5">ID: {face.roll_number || '—'} | CONF: {((face.confidence || 0) * 100).toFixed(0)}%</p>
                  </div>
                  <div className="shrink-0 w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_5px_rgba(52,211,153,0.5)]"></div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
