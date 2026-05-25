import { create } from 'zustand'

const BACKEND = 'http://localhost:5000'

/**
 * Global recording preference store.
 *
 * Why this exists:
 * - auto_record / manual_record were previously stored in LiveHub component state.
 * - When the user navigated away, the component unmounted and local state was lost.
 * - On return, the component re-mounted and re-synced from the backend, which
 *   could briefly show the wrong state or silently start recording again.
 *
 * This Zustand store lives for the entire app session (not tied to any component),
 * so the user's toggle choice is preserved across all page navigations.
 */
const useRecordingStore = create((set, get) => ({
  autoRecord: true,
  manualRecord: false,
  prefsLoaded: false,   // true once we've synced from the backend for the first time

  /** Called on first successful /status poll — only syncs if not yet loaded. */
  syncFromBackend: (statusData) => {
    if (get().prefsLoaded) return
    set({
      autoRecord: statusData.auto_record !== false,
      manualRecord: !!statusData.manual_record,
      prefsLoaded: true,
    })
  },

  /** Toggle auto-record. Sends request to backend; reverts on failure. */
  toggleAutoRecord: async () => {
    const prev = get().autoRecord
    const next = !prev
    set({ autoRecord: next })   // optimistic
    try {
      await fetch(`${BACKEND}/toggle-recording`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: next }),
      })
    } catch (e) {
      set({ autoRecord: prev })  // revert on network error
      console.error('[RecordingStore] toggle-auto-record failed:', e)
    }
  },

  /** Toggle manual record. Sends request to backend; reverts on failure. */
  toggleManualRecord: async () => {
    const prev = get().manualRecord
    const next = !prev
    set({ manualRecord: next })  // optimistic
    try {
      await fetch(`${BACKEND}/manual-record`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: next }),
      })
    } catch (e) {
      set({ manualRecord: prev }) // revert on network error
      console.error('[RecordingStore] toggle-manual-record failed:', e)
    }
  },
}))

export default useRecordingStore
