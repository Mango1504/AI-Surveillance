import { create } from 'zustand'

const useAuthStore = create((set) => ({
  user: JSON.parse(localStorage.getItem('user')) || null,
  isAuthenticated: !!localStorage.getItem('user'),

  login: (userData) => {
    localStorage.setItem('user', JSON.stringify(userData))
    set({
      user: userData,
      isAuthenticated: true,
    })
  },

  logout: () => {
    localStorage.removeItem('user')
    set({
      user: null,
      isAuthenticated: false,
    })
  },

  setUser: (userData) => {
    set({ user: userData })
  },
}))

export default useAuthStore
