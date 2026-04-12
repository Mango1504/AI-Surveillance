import { create } from 'zustand'

const useAlertStore = create((set, get) => ({
  alerts: [],
  
  addAlert: (alert) => {
    const newAlert = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      ...alert,
    }
    set((state) => ({
      alerts: [newAlert, ...state.alerts],
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
