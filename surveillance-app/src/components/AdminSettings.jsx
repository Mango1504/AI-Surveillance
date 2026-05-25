import { useState, useEffect, useRef, useCallback } from 'react'
import { Cpu, Database, Save, RotateCcw, ShieldAlert, Monitor, Terminal, Activity, Zap, PlayCircle, StopCircle, FlaskConical } from 'lucide-react'
import { apiService } from '../services/api'

export default function AdminSettings() {
  const [hudEnabled, setHudEnabled] = useState(true)
  const [yoloConf, setYoloConf] = useState(65)
  const [objectDetEnabled, setObjectDetEnabled] = useState(true)
  const [gazeEnabled, setGazeEnabled] = useState(true)
  const [detectEveryN, setDetectEveryN] = useState(2)
  const [resolution, setResolution] = useState('1080p')
  const [saving, setSaving] = useState(false)
  const [saveMsg, setSaveMsg] = useState('')

  // Risk Engine state (Layer 4)
  const [riskThreshold, setRiskThreshold] = useState(65)
  const [temporalWindow, setTemporalWindow] = useState(5)
  const [minDuration, setMinDuration] = useState(2)

  // Session state (DII Layer 2)
  const [sessionActive, setSessionActive] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [sessionCandidates, setSessionCandidates] = useState(0)
  const [sessionLoading, setSessionLoading] = useState(false)
  
  const [systemInfo, setSystemInfo] = useState(null)
  const [logs, setLogs] = useState([])
  const logContainerRef = useRef(null)   // ref on the scrollable div, NOT a sentinel element
  const userScrolledUp = useRef(false)   // true when user has scrolled up inside the log box

  const addLog = useCallback((type, msg) => {
    setLogs(prev => {
      const newLogs = [...prev, `[${new Date().toISOString()}] [${type}] ${msg}`]
      if (newLogs.length > 80) return newLogs.slice(newLogs.length - 80)
      return newLogs
    })
  }, [])

  // Load current config from backend on mount
  useEffect(() => {
    const loadConfig = async () => {
      try {
        const cfg = await apiService.getConfig()
        setYoloConf(cfg.yolo_confidence || 65)
        setObjectDetEnabled(cfg.object_det_enabled !== false)
        setGazeEnabled(cfg.gaze_enabled !== false)
        setDetectEveryN(cfg.detect_every_n || 2)
        if (cfg.resolution) {
          if (cfg.resolution.includes('3840')) setResolution('4k')
          else if (cfg.resolution.includes('1280')) setResolution('720p')
          else setResolution('1080p')
        }
      } catch {
        addLog('ERROR', 'Failed to load config from backend')
      }
    }
    loadConfig()
  }, [addLog])

  // Poll system info every 3 seconds
  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const info = await apiService.getSystemInfo()
        setSystemInfo(info)
      } catch {
        // Backend may be offline
      }
    }
    fetchInfo()
    const interval = setInterval(fetchInfo, 3000)
    return () => clearInterval(interval)
  }, [])

  // Real system logs from polling
  useEffect(() => {
    if (!systemInfo) return
    
    const tier = systemInfo.tier
    const cpu = systemInfo.cpu_percent
    const ram = systemInfo.ram_percent
    const dropRate = systemInfo.frame_drop_rate
    
    const logType = cpu > 80 ? 'WARN' : 'INFO'
    const msg = `[MONITOR] CPU: ${cpu?.toFixed(1)}% | RAM: ${ram?.toFixed(1)}% | Queue: ${systemInfo.frame_queue_size} | Drop: ${dropRate}% | Tier: ${tier}`
    
    addLog(logType, msg)
  }, [systemInfo, addLog])

  // Boot log
  useEffect(() => {
    const bootMessages = [
      '[SYSTEM] Admin panel connected to backend...',
      '[SYSTEM] Polling /system-info for live telemetry...',
    ]
    bootMessages.forEach((msg, i) => {
      setTimeout(() => {
        setLogs(prev => [...prev, `[${new Date().toISOString()}] ${msg}`])
      }, i * 500)
    })
  }, [])

  // Scroll the terminal box to the bottom when new logs arrive —
  // but ONLY if the user hasn't scrolled up to read old entries.
  // Uses scrollTop on the container div so the page itself never jumps.
  useEffect(() => {
    const el = logContainerRef.current
    if (!el || userScrolledUp.current) return
    el.scrollTop = el.scrollHeight
  }, [logs])

  const handleSave = async () => {
    setSaving(true)
    setSaveMsg('')
    try {
      await apiService.updateConfig({
        yolo_confidence: Number(yoloConf),
        object_det_enabled: objectDetEnabled,
        gaze_enabled: gazeEnabled,
        detect_every_n: Number(detectEveryN),
        risk_threshold: riskThreshold / 100,
        temporal_window: Number(temporalWindow),
        min_detection_duration: Number(minDuration),
      })
      setSaveMsg('✓ Config saved')
      addLog('INFO', `[CONFIG] Saved: YOLO=${yoloConf}%, Risk=${riskThreshold}%, Window=${temporalWindow}s, MinDur=${minDuration}s`)
    } catch {
      setSaveMsg('✗ Save failed')
      addLog('ERROR', '[CONFIG] Failed to save configuration to backend')
    } finally {
      setSaving(false)
      setTimeout(() => setSaveMsg(''), 3000)
    }
  }

  const handleSessionStart = async () => {
    setSessionLoading(true)
    try {
      const res = await fetch('http://localhost:5000/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidates: [] }),
      }).then(r => r.json())
      setSessionActive(true)
      setSessionId(res.session_id)
      setSessionCandidates(res.candidates_loaded || 0)
      addLog('INFO', `[SESSION] Started session #${res.session_id} | Candidates: ${res.candidates_loaded}`)
    } catch {
      addLog('ERROR', '[SESSION] Failed to start session')
    } finally {
      setSessionLoading(false)
    }
  }

  const handleSessionEnd = async () => {
    if (!window.confirm('End session? Non-flagged video data will be purged (≤5 min).')) return
    setSessionLoading(true)
    try {
      await fetch('http://localhost:5000/session/end', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
      setSessionActive(false)
      setSessionId(null)
      setSessionCandidates(0)
      addLog('WARN', '[SESSION] Session ended. Non-flagged data purge initiated.')
    } catch {
      addLog('ERROR', '[SESSION] Failed to end session')
    } finally {
      setSessionLoading(false)
    }
  }

  const handlePurge = async () => {
    if (!window.confirm('Purge all biometric embeddings from memory? This cannot be undone.')) return
    try {
      await apiService.purgeBiometrics()
      addLog('WARN', '[IDENTITY] All biometric vector embeddings purged from memory')
    } catch {
      addLog('ERROR', '[IDENTITY] Failed to purge biometrics')
    }
  }

  const handleRestart = () => {
    addLog('WARN', '[SYSTEM] Restart requested — please restart the backend manually via start_surveillance.bat')
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 @container">
      <div className="mb-6 flex justify-between items-end shrink-0">
        <div>
          <h2 className="text-3xl font-bold text-on-surface tracking-tight">System Configuration</h2>
          <p className="font-mono text-[13px] text-on-surface-variant mt-1">
            {systemInfo ? `Tier: ${systemInfo.tier} • ${systemInfo.cpu_cores}-core CPU • ${systemInfo.gpu} GPU` : 'Connecting to backend...'}
          </p>
        </div>
        <div className="flex gap-2 items-center">
          {saveMsg && (
            <span className={`font-mono text-xs ${saveMsg.includes('✓') ? 'text-emerald-400' : 'text-error'}`}>
              {saveMsg}
            </span>
          )}
          <button onClick={handleRestart} className="bg-surface-container text-on-surface font-mono text-sm px-4 py-2 border border-outline-variant/50 rounded hover:bg-surface-container-high transition-colors flex items-center gap-2">
            <RotateCcw size={16} /> RESTART CORE
          </button>
          <button onClick={handleSave} disabled={saving} className="bg-primary text-on-primary font-mono text-sm px-4 py-2 rounded hover:bg-primary-fixed transition-colors flex items-center gap-2 shadow-[0_0_15px_rgba(173,198,255,0.2)] disabled:opacity-50">
            <Save size={16} /> {saving ? 'SAVING...' : 'SAVE CONFIG'}
          </button>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 overflow-y-auto pb-6">
        
        {/* Left Column */}
        <div className="flex flex-col gap-6">
          {/* AI Model Tuning */}
          <section className="bg-surface-container border border-outline-variant/30 rounded-xl overflow-hidden shadow-lg">
            <div className="p-4 border-b border-outline-variant/30 bg-surface-container-low flex items-center gap-2 text-on-surface">
              <Cpu size={18} className="text-primary" />
              <h3 className="font-bold uppercase tracking-wide text-sm">AI Model Tuning</h3>
            </div>
            <div className="p-6 flex flex-col gap-6">
              <div>
                <div className="flex justify-between items-end mb-2">
                  <label className="font-mono text-xs text-on-surface-variant uppercase">YOLO Confidence Threshold</label>
                  <span className="font-mono text-sm font-bold text-primary">{yoloConf}%</span>
                </div>
                <input 
                  type="range" 
                  min="30" max="95" 
                  value={yoloConf} 
                  onChange={(e) => setYoloConf(e.target.value)}
                />
              </div>
              <div>
                <div className="flex justify-between items-end mb-2">
                  <label className="font-mono text-xs text-on-surface-variant uppercase">Frame Skip (Detect Every N)</label>
                  <span className="font-mono text-sm font-bold text-primary">{detectEveryN}</span>
                </div>
                <input 
                  type="range" 
                  min="1" max="10" step="1"
                  value={detectEveryN} 
                  onChange={(e) => setDetectEveryN(e.target.value)}
                />
              </div>
              <div className="flex justify-between items-center">
                <div className="flex flex-col">
                  <span className="font-body-md text-on-surface font-semibold">Object Detection (YOLO)</span>
                  <span className="font-mono text-[11px] text-on-surface-variant mt-1">Enable/disable real-time phone detection</span>
                </div>
                <div 
                  className={`toggle-track ${objectDetEnabled ? 'active' : ''}`}
                  onClick={() => setObjectDetEnabled(!objectDetEnabled)}
                ></div>
              </div>
              <div className="flex justify-between items-center">
                <div className="flex flex-col">
                  <span className="font-body-md text-on-surface font-semibold">Gaze Tracking</span>
                  <span className="font-mono text-[11px] text-on-surface-variant mt-1">Track student head/eye direction</span>
                </div>
                <div 
                  className={`toggle-track ${gazeEnabled ? 'active' : ''}`}
                  onClick={() => setGazeEnabled(!gazeEnabled)}
                ></div>
              </div>
            </div>
          </section>

          {/* Identity Database + Session Controls */}
          <section className="bg-surface-container border border-outline-variant/30 rounded-xl overflow-hidden shadow-lg">
            <div className="p-4 border-b border-outline-variant/30 bg-surface-container-low flex items-center gap-2 text-on-surface">
              <Database size={18} className="text-primary" />
              <h3 className="font-bold uppercase tracking-wide text-sm">Identity & Session</h3>
            </div>
            <div className="p-6 flex flex-col gap-4">
              {/* Session status badge */}
              <div className="flex justify-between items-center bg-surface-container-low p-4 rounded border border-outline-variant/20">
                <div className="flex flex-col">
                  <span className="font-mono text-xs text-on-surface-variant uppercase">Session</span>
                  <span className={`font-mono text-xs font-bold ${sessionActive ? 'text-emerald-400' : 'text-on-surface-variant'}`}>
                    {sessionActive ? `ACTIVE · #${sessionId} · ${sessionCandidates} enrolled` : 'INACTIVE'}
                  </span>
                </div>
                <span className="font-mono text-xs text-on-surface-variant">VGG-Face / FAR &lt;0.01%</span>
              </div>

              {/* Session Start / End */}
              <div className="flex gap-3">
                <button
                  id="session-start-btn"
                  onClick={handleSessionStart}
                  disabled={sessionActive || sessionLoading}
                  className="flex-1 bg-emerald-600/20 text-emerald-400 border border-emerald-500/50 hover:bg-emerald-600/40 font-mono text-xs py-2.5 rounded transition-colors flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <PlayCircle size={15} /> START SESSION
                </button>
                <button
                  id="session-end-btn"
                  onClick={handleSessionEnd}
                  disabled={!sessionActive || sessionLoading}
                  className="flex-1 bg-yellow-600/20 text-yellow-400 border border-yellow-500/50 hover:bg-yellow-600/40 font-mono text-xs py-2.5 rounded transition-colors flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <StopCircle size={15} /> END SESSION
                </button>
              </div>

              <button onClick={handlePurge} className="w-full bg-error-container/20 text-error border border-error/50 hover:bg-error-container hover:text-on-error-container font-mono text-xs py-2.5 rounded transition-colors flex items-center justify-center gap-2">
                <ShieldAlert size={16} /> PURGE VECTOR MEMORY
              </button>
            </div>
          </section>

          {/* Risk Engine Tuning (Layer 4) */}
          <section className="bg-surface-container border border-outline-variant/30 rounded-xl overflow-hidden shadow-lg">
            <div className="p-4 border-b border-outline-variant/30 bg-surface-container-low flex items-center gap-2 text-on-surface">
              <FlaskConical size={18} className="text-primary" />
              <h3 className="font-bold uppercase tracking-wide text-sm">Risk Engine · Layer 4</h3>
            </div>
            <div className="p-6 flex flex-col gap-5">
              <div>
                <div className="flex justify-between items-end mb-2">
                  <label className="font-mono text-xs text-on-surface-variant uppercase">Alert Threshold</label>
                  <span className="font-mono text-sm font-bold text-primary">{riskThreshold}%</span>
                </div>
                <input type="range" min="10" max="95" value={riskThreshold} onChange={e => setRiskThreshold(+e.target.value)} />
                <p className="font-mono text-[10px] text-on-surface-variant mt-1">Min composite risk score to fire an incident (paper: 0.65)</p>
              </div>
              <div>
                <div className="flex justify-between items-end mb-2">
                  <label className="font-mono text-xs text-on-surface-variant uppercase">Temporal Window</label>
                  <span className="font-mono text-sm font-bold text-primary">{temporalWindow}s</span>
                </div>
                <input type="range" min="1" max="15" step="1" value={temporalWindow} onChange={e => setTemporalWindow(+e.target.value)} />
                <p className="font-mono text-[10px] text-on-surface-variant mt-1">Sliding window for score accumulation</p>
              </div>
              <div>
                <div className="flex justify-between items-end mb-2">
                  <label className="font-mono text-xs text-on-surface-variant uppercase">Min Detection Duration</label>
                  <span className="font-mono text-sm font-bold text-primary">{minDuration}s</span>
                </div>
                <input type="range" min="0.5" max="10" step="0.5" value={minDuration} onChange={e => setMinDuration(+e.target.value)} />
                <p className="font-mono text-[10px] text-on-surface-variant mt-1">Object must be visible ≥ this long before alert fires (D-03)</p>
              </div>
            </div>
          </section>
          
          {/* Live Telemetry */}
          {systemInfo && (
            <section className="bg-surface-container border border-outline-variant/30 rounded-xl overflow-hidden shadow-lg">
              <div className="p-4 border-b border-outline-variant/30 bg-surface-container-low flex items-center gap-2 text-on-surface">
                <Activity size={18} className="text-primary" />
                <h3 className="font-bold uppercase tracking-wide text-sm">Live Telemetry</h3>
              </div>
              <div className="p-4 grid grid-cols-2 gap-3">
                <div className="bg-surface-container-low rounded p-3 border border-outline-variant/20">
                  <div className="font-mono text-[10px] text-on-surface-variant uppercase mb-1">CPU</div>
                  <div className="font-bold text-lg text-on-surface">{systemInfo.cpu_percent?.toFixed(1)}%</div>
                  <div className="w-full h-1 bg-surface-container-highest rounded mt-1">
                    <div className={`h-1 rounded transition-all ${systemInfo.cpu_percent > 80 ? 'bg-error' : systemInfo.cpu_percent > 60 ? 'bg-yellow-500' : 'bg-emerald-500'}`} style={{ width: `${systemInfo.cpu_percent}%` }}></div>
                  </div>
                </div>
                <div className="bg-surface-container-low rounded p-3 border border-outline-variant/20">
                  <div className="font-mono text-[10px] text-on-surface-variant uppercase mb-1">RAM</div>
                  <div className="font-bold text-lg text-on-surface">{systemInfo.ram_percent?.toFixed(1)}%</div>
                  <div className="w-full h-1 bg-surface-container-highest rounded mt-1">
                    <div className={`h-1 rounded transition-all ${systemInfo.ram_percent > 85 ? 'bg-error' : systemInfo.ram_percent > 70 ? 'bg-yellow-500' : 'bg-emerald-500'}`} style={{ width: `${systemInfo.ram_percent}%` }}></div>
                  </div>
                </div>
                <div className="bg-surface-container-low rounded p-3 border border-outline-variant/20">
                  <div className="font-mono text-[10px] text-on-surface-variant uppercase mb-1">Frame Queue</div>
                  <div className="font-bold text-lg text-on-surface">{systemInfo.frame_queue_size}</div>
                  <div className="font-mono text-[10px] text-on-surface-variant">Drop: {systemInfo.frame_drop_rate}%</div>
                </div>
                <div className="bg-surface-container-low rounded p-3 border border-outline-variant/20">
                  <div className="font-mono text-[10px] text-on-surface-variant uppercase mb-1">Workers</div>
                  <div className="font-bold text-lg text-on-surface flex items-center gap-2">
                    {systemInfo.num_workers}
                    <Zap size={14} className="text-primary" />
                  </div>
                  <div className="font-mono text-[10px] text-on-surface-variant">{systemInfo.model_name}</div>
                </div>
              </div>
            </section>
          )}

          {/* Hardware Interface */}
          <section className="bg-surface-container border border-outline-variant/30 rounded-xl overflow-hidden shadow-lg">
            <div className="p-4 border-b border-outline-variant/30 bg-surface-container-low flex items-center gap-2 text-on-surface">
              <Monitor size={18} className="text-primary" />
              <h3 className="font-bold uppercase tracking-wide text-sm">Hardware Interface</h3>
            </div>
            <div className="p-6 flex flex-col gap-6">
              <div className="flex justify-between items-center">
                <div className="flex flex-col">
                  <span className="font-body-md text-on-surface font-semibold">Enable HUD Grid Overlay</span>
                  <span className="font-mono text-[11px] text-on-surface-variant mt-1">Displays spatial tracking grid on live feed</span>
                </div>
                <div 
                  className={`toggle-track ${hudEnabled ? 'active' : ''}`}
                  onClick={() => setHudEnabled(!hudEnabled)}
                ></div>
              </div>
              
              <div className="flex justify-between items-center">
                <div className="flex flex-col">
                  <span className="font-body-md text-on-surface font-semibold">Capture Resolution</span>
                  <span className="font-mono text-[11px] text-on-surface-variant mt-1">Video stream quality from source</span>
                </div>
                <select 
                  className="bg-surface-container-low border border-outline-variant/50 text-on-surface font-mono text-xs p-2 rounded focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                  value={resolution}
                  onChange={(e) => setResolution(e.target.value)}
                >
                  <option value="720p">720p (HD)</option>
                  <option value="1080p">1080p (FHD)</option>
                  <option value="4k">4K (UHD)</option>
                </select>
              </div>
            </div>
          </section>
        </div>

        {/* Right Column - Terminal Logs */}
        <section className="bg-[#020617] border border-outline-variant/30 rounded-xl overflow-hidden shadow-lg flex flex-col h-full min-h-[500px]">
          <div className="p-4 border-b border-outline-variant/30 bg-[#0f172a] flex justify-between items-center shrink-0">
            <div className="flex items-center gap-2 text-emerald-400">
              <Terminal size={18} />
              <h3 className="font-mono font-bold uppercase tracking-wide text-sm">System Kernel Log</h3>
            </div>
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-error border border-error-container"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500 border border-yellow-600"></div>
              <div className="w-3 h-3 rounded-full bg-emerald-500 border border-emerald-600"></div>
            </div>
          </div>
          <div
            ref={logContainerRef}
            className="flex-1 overflow-y-auto p-4 font-mono text-[11px] leading-relaxed relative"
            onScroll={() => {
              const el = logContainerRef.current
              if (!el) return
              // If user scrolls more than 60px from the bottom, pause auto-scroll
              const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
              userScrolledUp.current = distFromBottom > 60
            }}
          >
            <div className="absolute top-0 left-0 w-full h-4 bg-gradient-to-b from-[#020617] to-transparent sticky z-10 pointer-events-none"></div>
            {logs.map((log, i) => {
              let colorClass = 'text-outline'
              if (log.includes('[WARN]')) colorClass = 'text-yellow-400'
              if (log.includes('[ERROR]')) colorClass = 'text-error'
              if (log.includes('[SYSTEM]')) colorClass = 'text-primary'
              if (log.includes('[CONFIG]')) colorClass = 'text-[#a78bfa]'
              if (log.includes('[IDENTITY]')) colorClass = 'text-[#a78bfa]'
              if (log.includes('[MONITOR]')) colorClass = 'text-emerald-400/70'
              
              return (
                <div key={i} className={`mb-1 hover:bg-white/5 px-1 rounded transition-colors ${colorClass}`}>
                  {log}
                </div>
              )
            })}
          </div>
        </section>
        
      </div>
    </div>
  )
}
