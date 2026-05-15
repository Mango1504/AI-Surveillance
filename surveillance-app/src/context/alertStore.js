import { create } from 'zustand'

const useAlertStore = create((set, get) => ({
  alerts: [],
  
  addAlert: (alert) => {
    const now = Date.now()
    const existing = get().alerts

    // Deduplicate: skip if same grid cell was alerted within 5 seconds
    const isDuplicate = existing.some(
      (a) =>
        a.row === alert.row &&
        a.col === alert.col &&
        a.examHall === alert.examHall &&
        (now - a.id) < 5000
    )
    if (isDuplicate) return

    const newAlert = {
      id: now,
      timestamp: new Date().toISOString(),
      ...alert,
    }
    set((state) => ({
      // Cap at 50 entries, newest first
      alerts: [newAlert, ...state.alerts].slice(0, 50),
    }))
  },

  clearAlerts: () => set({ alerts: [] }),

  deleteAlert: (id) => {
    set((state) => ({
      alerts: state.alerts.filter((a) => a.id !== id),
    }))
  },

  getAlerts: () => get().alerts,
}))

export default useAlertStore
