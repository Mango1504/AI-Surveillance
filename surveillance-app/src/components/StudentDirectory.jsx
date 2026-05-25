import { useState, useEffect, useRef } from 'react'
import { Search, SlidersHorizontal, ChevronRight, Verified, BookOpen, Calendar, MapPin, Flag, Video, Upload, AlertOctagon, UserX, CheckCircle, XCircle } from 'lucide-react'
import { apiService } from '../services/api'

export default function StudentDirectory() {
  const [tab, setTab] = useState('registry') // 'registry' | 'enroll' | 'intruders'
  const [applicants, setApplicants] = useState([])
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedPerson, setSelectedPerson] = useState(null)
  const [loading, setLoading] = useState(true)

  // Enroll state
  const [enrollFile, setEnrollFile] = useState(null)
  const [enrollPreview, setEnrollPreview] = useState(null)
  const [enrollRoll, setEnrollRoll] = useState('')
  const [enrollName, setEnrollName] = useState('')
  const [enrolling, setEnrolling] = useState(false)
  const [enrollMsg, setEnrollMsg] = useState(null) // {ok, text}
  const fileInputRef = useRef(null)

  // Intruders state
  const [intruders, setIntruders] = useState([])
  const [intrudersLoading, setIntrudersLoading] = useState(false)

  const fetchIntruders = async () => {
    setIntrudersLoading(true)
    try {
      const res = await fetch('http://localhost:5000/intruders')
      const data = await res.json()
      setIntruders(data)
    } catch { setIntruders([]) }
    finally { setIntrudersLoading(false) }
  }

  useEffect(() => {
    if (tab === 'intruders') fetchIntruders()
  }, [tab])

  const handleFileChange = (e) => {
    const f = e.target.files[0]
    if (!f) return
    setEnrollFile(f)
    setEnrollPreview(URL.createObjectURL(f))
    setEnrollMsg(null)
  }

  const handleEnroll = async (e) => {
    e.preventDefault()
    if (!enrollFile || !enrollRoll.trim()) return
    setEnrolling(true)
    setEnrollMsg(null)
    try {
      const fd = new FormData()
      fd.append('file', enrollFile)
      fd.append('roll_number', enrollRoll.trim())
      fd.append('name', enrollName.trim() || enrollRoll.trim())
      const res = await fetch('http://localhost:5000/enroll', { method: 'POST', body: fd })
      const data = await res.json()
      if (data.success) {
        setEnrollMsg({ ok: true, text: data.message })
        setEnrollFile(null); setEnrollPreview(null); setEnrollRoll(''); setEnrollName('')
      } else {
        setEnrollMsg({ ok: false, text: data.error || 'Enrollment failed' })
      }
    } catch (err) {
      setEnrollMsg({ ok: false, text: 'Network error — backend offline?' })
    } finally { setEnrolling(false) }
  }

  useEffect(() => {
    const fetchApplicants = async () => {
      setLoading(true)
      try {
        const data = await apiService.getApplicantsInfo()
        setApplicants(data.applicants || [])
        if (data.applicants?.length > 0) setSelectedPerson(data.applicants[0])
      } catch (err) { console.error('Failed to load applicants', err) }
      finally { setLoading(false) }
    }
    fetchApplicants()
  }, [])

  const filteredApplicants = applicants.filter((person) => {
    const rollNumber = person.roll_number || ''
    const name = person.info?.name || ''
    const term = searchTerm.toLowerCase()
    return rollNumber.toLowerCase().includes(term) || name.toLowerCase().includes(term)
  })

  return (
    <div className="flex-1 flex flex-col min-h-0 @container">

      {/* Tab Bar */}
      <div className="flex gap-1 mb-4 shrink-0 bg-surface-container-low rounded-xl p-1 border border-outline-variant/20">
        {[
          { key: 'registry', label: 'Candidate Registry' },
          { key: 'enroll',   label: 'Enroll via ID Card' },
          { key: 'intruders', label: 'Intruder Log' },
        ].map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
            className={`flex-1 py-2 rounded-lg font-mono text-xs uppercase tracking-widest transition-all ${
              tab === t.key
                ? 'bg-primary text-on-primary shadow-[0_0_10px_rgba(173,198,255,0.2)]'
                : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
            }`}>{t.label}
          </button>
        ))}
      </div>

      {/* ── ENROLL TAB ── */}
      {tab === 'enroll' && (
        <div className="flex-1 flex flex-col lg:flex-row gap-6 overflow-y-auto min-h-0">
          <form onSubmit={handleEnroll} className="flex-1 bg-surface-container border border-outline-variant/30 rounded-xl p-6 flex flex-col gap-5">
            <div>
              <h2 className="text-xl font-bold text-on-surface mb-1">Enroll Authorized Person</h2>
              <p className="text-xs font-mono text-on-surface-variant">Upload admit card or ID photo. The system extracts the face and adds it to the authorized identity database.</p>
            </div>

            {/* ID Card Upload */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-outline-variant/60 rounded-xl p-6 flex flex-col items-center justify-center gap-3 cursor-pointer hover:border-primary/60 hover:bg-primary/5 transition-all min-h-[180px]"
            >
              {enrollPreview ? (
                <img src={enrollPreview} alt="ID Card Preview" className="max-h-40 rounded-lg object-contain border border-outline-variant/30" />
              ) : (
                <>
                  <Upload size={36} className="text-on-surface-variant opacity-50" />
                  <p className="font-mono text-xs text-on-surface-variant text-center">Click to upload ID Card / Admit Card<br/><span className="text-[10px] opacity-60">JPEG, PNG supported</span></p>
                </>
              )}
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
            </div>

            {/* Roll Number */}
            <div>
              <label className="block font-mono text-[11px] text-on-surface-variant uppercase tracking-wider mb-1">Roll Number / ID *</label>
              <input
                type="text" value={enrollRoll} onChange={e => setEnrollRoll(e.target.value)} required
                placeholder="e.g. 2024CS001"
                className="w-full bg-surface-container-low border border-outline-variant/60 rounded-lg px-4 py-2 font-mono text-sm text-on-surface placeholder:text-on-surface-variant/40 focus:outline-none focus:border-primary transition-colors"
              />
            </div>

            {/* Name */}
            <div>
              <label className="block font-mono text-[11px] text-on-surface-variant uppercase tracking-wider mb-1">Full Name (optional)</label>
              <input
                type="text" value={enrollName} onChange={e => setEnrollName(e.target.value)}
                placeholder="e.g. Arjun Sharma"
                className="w-full bg-surface-container-low border border-outline-variant/60 rounded-lg px-4 py-2 font-mono text-sm text-on-surface placeholder:text-on-surface-variant/40 focus:outline-none focus:border-primary transition-colors"
              />
            </div>

            {/* Feedback */}
            {enrollMsg && (
              <div className={`flex items-center gap-2 px-4 py-3 rounded-lg border text-sm font-mono ${
                enrollMsg.ok
                  ? 'bg-emerald-900/20 border-emerald-500/40 text-emerald-400'
                  : 'bg-red-900/20 border-red-500/40 text-red-400'
              }`}>
                {enrollMsg.ok ? <CheckCircle size={16}/> : <XCircle size={16}/>}
                {enrollMsg.text}
              </div>
            )}

            <button type="submit" disabled={enrolling || !enrollFile || !enrollRoll.trim()}
              className="w-full py-3 bg-primary text-on-primary font-mono text-sm rounded-lg hover:bg-primary-fixed transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {enrolling ? (
                <><svg className="animate-spin" width="18" height="18" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/></svg>Enrolling...</>
              ) : (
                <><Upload size={16}/>Enroll &amp; Save to Identity DB</>
              )}
            </button>
          </form>

          {/* Info Panel */}
          <div className="w-full lg:w-72 flex flex-col gap-4 shrink-0">
            <div className="bg-surface-container border border-outline-variant/30 rounded-xl p-5 flex flex-col gap-3">
              <h3 className="font-mono text-xs text-on-surface-variant uppercase tracking-widest">How It Works</h3>
              <div className="flex flex-col gap-3 text-xs text-on-surface-variant font-mono">
                {[
                  ['1', 'Upload admit card or photo ID with a visible face'],
                  ['2', 'System auto-extracts the face using DeepFace'],
                  ['3', 'Face embedding saved to authorized identity DB'],
                  ['4', 'Live feed now marks this person as AUTHORIZED'],
                  ['5', 'Anyone not in DB → flagged as INTRUDER'],
                ].map(([n, t]) => (
                  <div key={n} className="flex gap-3">
                    <span className="w-5 h-5 rounded-full bg-primary/20 text-primary flex items-center justify-center text-[10px] shrink-0">{n}</span>
                    <span>{t}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="bg-amber-900/10 border border-amber-500/30 rounded-xl p-4">
              <p className="font-mono text-[11px] text-amber-400"><strong>Note:</strong> A session must be active (Admin Panel → Start Session) for intruder detection to run.</p>
            </div>
          </div>
        </div>
      )}

      {/* ── INTRUDERS TAB ── */}
      {tab === 'intruders' && (
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex justify-between items-center mb-4 shrink-0">
            <div>
              <h2 className="text-xl font-bold text-on-surface">Intruder Log</h2>
              <p className="font-mono text-[12px] text-on-surface-variant mt-0.5">Unauthorized persons detected and stored in Identity DB</p>
            </div>
            <button onClick={fetchIntruders} className="font-mono text-xs px-3 py-1.5 rounded border border-outline-variant text-on-surface-variant hover:bg-surface-container-high transition-colors">Refresh</button>
          </div>
          <div className="flex-1 overflow-y-auto min-h-0">
            {intrudersLoading ? (
              <div className="flex items-center justify-center h-32 text-on-surface-variant font-mono text-sm">Loading...</div>
            ) : intruders.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-48 gap-3 text-on-surface-variant">
                <UserX size={40} className="opacity-30"/>
                <p className="font-mono text-sm">No intruders logged yet</p>
                <p className="font-mono text-[11px] opacity-50">Detections appear here when unauthorized persons are found</p>
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                {intruders.map(r => (
                  <div key={r.id} className="bg-surface-container border border-red-900/30 rounded-lg p-4 flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-red-900/30 border border-red-500/40 flex items-center justify-center shrink-0">
                      <AlertOctagon size={18} className="text-red-400"/>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-mono text-xs font-bold text-red-400 uppercase tracking-wide">UNAUTHORIZED PERSON</p>
                      <p className="font-mono text-[11px] text-on-surface-variant mt-0.5">
                        Camera {r.camera_id} &nbsp;•&nbsp;
                        {new Date(r.timestamp * 1000).toLocaleString(undefined, { hour12: false })}
                      </p>
                      {r.notes && <p className="font-mono text-[10px] text-on-surface-variant/60 mt-0.5 truncate">{r.notes}</p>}
                    </div>
                    <div className="shrink-0 font-mono text-xs text-on-surface-variant">
                      Conf: {r.confidence ? (r.confidence * 100).toFixed(0) : '—'}%
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── REGISTRY TAB ── */}
      {tab === 'registry' && (
        <div className="flex-1 flex flex-col lg:flex-row overflow-hidden gap-6 h-full min-h-0">
      
      {/* Left Pane: Registry List */}
      <section className="flex-1 flex flex-col min-w-0 bg-surface-container-lowest rounded-xl border border-outline-variant/20 overflow-hidden shadow-lg min-h-0">
        
        {/* Search Header */}
        <div className="p-4 border-b border-outline-variant/30 bg-surface-container-low shrink-0 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-on-surface">Candidate Registry</h2>
            <div className="flex items-center gap-2">
              <span className="font-mono text-[11px] text-on-surface-variant uppercase tracking-wider">Active Entities: {applicants.length}</span>
            </div>
          </div>
          <div className="relative w-full">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-on-surface-variant" size={18} />
            <input
              type="text"
              placeholder="Search by ID or Name..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-surface-container border border-outline-variant text-on-surface py-2.5 pl-12 pr-4 rounded-lg focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary placeholder-on-surface-variant/60 transition-all text-sm"
            />
            <button className="absolute right-3 top-1/2 -translate-y-1/2 text-on-surface-variant hover:text-on-surface p-1 rounded hover:bg-surface-container-high transition-colors">
              <SlidersHorizontal size={16} />
            </button>
          </div>
        </div>

        {/* Database List */}
        <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-2 min-h-0">
          {/* Table Header */}
          <div className="hidden sm:grid grid-cols-[40px_1fr_120px_160px_32px] items-center gap-4 px-3 py-2 border-b border-outline-variant/20 mb-2 sticky top-0 bg-surface-container-lowest z-10 shrink-0">
            <span className="font-mono text-[11px] text-on-surface-variant uppercase">Bio</span>
            <span className="font-mono text-[11px] text-on-surface-variant uppercase">Identity</span>
            <span className="font-mono text-[11px] text-on-surface-variant uppercase">Roll No.</span>
            <span className="font-mono text-[11px] text-on-surface-variant uppercase">Location</span>
            <span></span>
          </div>

          {loading ? (
            <div className="flex items-center justify-center p-8 text-on-surface-variant font-mono text-sm">Loading registry...</div>
          ) : filteredApplicants.map((person) => {
            const isSelected = selectedPerson?.roll_number === person.roll_number
            return (
              <div
                key={person.roll_number}
                onClick={() => setSelectedPerson(person)}
                className={`flex sm:grid sm:grid-cols-[40px_1fr_120px_160px_32px] items-center gap-4 p-3 rounded cursor-pointer transition-colors group ${
                  isSelected 
                    ? 'bg-surface-container-high border-l-2 border-primary border-y border-r border-outline-variant/20 shadow-[0_0_15px_rgba(173,198,255,0.05)]' 
                    : 'bg-surface-container-low hover:bg-surface-container border border-transparent hover:border-outline-variant/30'
                }`}
              >
                <div className={`w-10 h-10 rounded bg-surface-container-highest border ${isSelected ? 'border-primary/50' : 'border-outline-variant'} flex items-center justify-center shrink-0`}>
                  <span className="font-mono text-xs text-on-surface-variant">
                    {person.info?.name?.substring(0,2).toUpperCase()}
                  </span>
                </div>
                
                <div className="flex flex-col min-w-0 flex-1 sm:flex-none">
                  <span className={`font-semibold text-sm truncate ${isSelected ? 'text-on-surface' : 'text-on-surface-variant group-hover:text-on-surface'}`}>
                    {person.info?.name}
                  </span>
                  <span className="font-mono text-[11px] text-tertiary truncate">
                    {person.info?.department || 'Unknown Dept'}
                  </span>
                </div>
                
                <span className={`hidden sm:block font-mono text-xs tracking-wide ${isSelected ? 'text-primary' : 'text-outline'}`}>
                  {person.roll_number}
                </span>
                
                <div className="hidden sm:flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${isSelected ? 'bg-primary animate-pulse' : 'bg-outline-variant'}`}></span>
                  <span className="font-mono text-[11px] text-on-surface-variant truncate">
                    Hall {person.info?.exam_hall || 'Unknown'}
                  </span>
                </div>
                
                <ChevronRight size={18} className={`hidden sm:block ml-auto ${isSelected ? 'text-primary' : 'text-outline-variant group-hover:text-on-surface'}`} />
              </div>
            )
          })}
        </div>
      </section>

      {/* Right Pane: Profile Bento */}
      <aside className="w-full lg:w-[400px] flex flex-col gap-4 shrink-0 overflow-y-auto pb-4 lg:pb-0 min-h-0">
        {selectedPerson ? (
          <>
            {/* ID Card */}
            <div className="bg-surface-container border border-outline-variant/40 rounded-xl overflow-hidden shadow-lg flex flex-col relative shrink-0">
              <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(#adc6ff 1px, transparent 1px)', backgroundSize: '16px 16px' }}></div>
              <div className="p-6 flex flex-col items-center relative z-10">
                <div className="absolute top-4 right-4 flex items-center gap-1.5 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 px-2.5 py-1 rounded-full backdrop-blur-md">
                  <Verified size={14} />
                  <span className="font-mono text-[10px] tracking-wider uppercase font-bold">Verified Match</span>
                </div>
                
                <div className="w-24 h-24 rounded-full bg-surface-container-lowest border-4 border-surface-container-lowest shadow-[0_0_20px_rgba(173,198,255,0.15)] overflow-hidden mb-4 relative mt-2 flex items-center justify-center">
                  <span className="text-3xl text-outline-variant">{selectedPerson.info?.name?.charAt(0)}</span>
                  <div className="absolute top-0 left-0 w-full h-[2px] bg-primary/50 shadow-[0_0_8px_#adc6ff] animate-scan"></div>
                </div>
                
                <h3 className="text-xl font-bold text-on-surface mb-1">{selectedPerson.info?.name}</h3>
                <div className="bg-surface-container-highest px-3 py-1 rounded font-mono text-sm text-primary tracking-widest border border-outline-variant/50 shadow-inner">
                  {selectedPerson.roll_number}
                </div>
              </div>
            </div>

            {/* Academic Telemetry Bento */}
            <div className="grid grid-cols-2 gap-2 shrink-0">
              <div className="bg-surface-container-low border border-outline-variant/30 rounded-lg p-4 flex flex-col gap-1 relative overflow-hidden group">
                <BookOpen className="absolute -right-2 -top-2 text-outline/10 group-hover:text-primary/10 transition-colors" size={64} />
                <span className="font-mono text-[10px] text-on-surface-variant uppercase tracking-wider relative z-10">Department</span>
                <span className="font-semibold text-on-surface relative z-10 text-sm truncate">{selectedPerson.info?.department || 'N/A'}</span>
              </div>
              <div className="bg-surface-container-low border border-outline-variant/30 rounded-lg p-4 flex flex-col gap-1 relative overflow-hidden group">
                <Calendar className="absolute -right-2 -top-2 text-outline/10 group-hover:text-primary/10 transition-colors" size={64} />
                <span className="font-mono text-[10px] text-on-surface-variant uppercase tracking-wider relative z-10">Enrollment</span>
                <span className="font-semibold text-on-surface relative z-10 text-sm truncate">{selectedPerson.info?.enrollment_year || 'N/A'}</span>
              </div>
              <div className="col-span-2 bg-surface-container-low border border-outline-variant/30 rounded-lg p-4 flex flex-col gap-2 shrink-0">
                <span className="font-mono text-[10px] text-on-surface-variant uppercase tracking-wider">Current Assessment</span>
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-on-surface text-sm truncate">{selectedPerson.info?.subject}</span>
                  <span className="bg-surface-container-highest border border-outline-variant text-tertiary px-2 py-0.5 rounded text-[11px] font-mono shrink-0">
                    EXAM
                  </span>
                </div>
              </div>
            </div>

            {/* Spatial Tracking Module */}
            <div className="bg-surface-container border border-outline-variant/40 rounded-xl flex flex-col overflow-hidden relative shrink-0">
              <div className="h-[2px] w-full bg-gradient-to-r from-transparent via-primary/50 to-transparent"></div>
              <div className="p-3 border-b border-outline-variant/20 flex items-center justify-between bg-surface-container-low">
                <div className="flex items-center gap-2 text-on-surface">
                  <MapPin size={16} className="text-primary" />
                  <span className="font-semibold text-xs uppercase tracking-wide">Spatial Tracking</span>
                </div>
                <span className="flex h-2 w-2 relative">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                </span>
              </div>
              <div className="p-4 flex flex-col gap-4">
                <div className="flex justify-between items-end border-b border-outline-variant/20 pb-3">
                  <div className="flex flex-col gap-1">
                    <span className="font-mono text-[10px] text-on-surface-variant">Last Known Node</span>
                    <span className="font-semibold text-sm text-on-surface flex items-center gap-2">
                      Hall {selectedPerson.info?.exam_hall}
                    </span>
                  </div>
                </div>
                
                <div className="flex flex-col gap-3 relative before:absolute before:inset-y-2 before:left-[11px] before:w-[2px] before:bg-outline-variant/30">
                  <div className="flex gap-4 relative z-10">
                    <div className="w-6 h-6 rounded-full bg-surface-container-highest border border-primary flex items-center justify-center shrink-0 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-primary"></div>
                    </div>
                    <div className="flex flex-col">
                      <span className="font-mono text-[10px] text-primary">LIVE</span>
                      <span className="text-xs text-on-surface">Assigned to Examination Hall</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Action Footer */}
            <div className="flex gap-3 mt-auto shrink-0">
              <button className="flex-1 bg-surface-container border border-outline-variant hover:bg-surface-container-high hover:border-outline text-on-surface py-2 rounded-lg font-mono text-xs tracking-wider transition-colors flex items-center justify-center gap-2 shadow-sm">
                <Flag size={16} /> FLAG ENTITY
              </button>
              <button className="flex-1 bg-primary text-on-primary hover:bg-primary-fixed py-2 rounded-lg font-mono text-xs tracking-wider transition-colors flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(173,198,255,0.2)]">
                <Video size={16} fill="currentColor" /> VIEW FEED
              </button>
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-full bg-surface-container-low rounded-xl border border-outline-variant/20 text-on-surface-variant font-mono text-sm">
            Select an entity to view telemetry
          </div>
        )}
      </aside>
        </div>
      )}
    </div>
  )
}
