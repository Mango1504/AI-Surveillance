import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Eye, EyeOff, Shield, UserPlus } from 'lucide-react'
import useAuthStore from '../context/authStore'

export default function AdminPanel() {
  const [isLoginMode, setIsLoginMode] = useState(true)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const navigate = useNavigate()
  const { login } = useAuthStore()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setSuccess('')
    setIsLoading(true)

    if (isLoginMode) {
      // Admin Login
      if (username.trim() && password.trim()) {
        setTimeout(() => {
          login({
            username,
            isAdmin: true,
            loginTime: new Date().toISOString(),
          })
          setIsLoading(false)
          navigate('/dashboard')
        }, 500)
      } else {
        setError('Please enter username and password')
        setIsLoading(false)
      }
    } else {
      // Admin Signup
      if (!username.trim()) {
        setError('Username is required')
        setIsLoading(false)
        return
      }
      if (!password.trim()) {
        setError('Password is required')
        setIsLoading(false)
        return
      }
      if (password !== confirmPassword) {
        setError('Passwords do not match')
        setIsLoading(false)
        return
      }
      if (password.length < 6) {
        setError('Password must be at least 6 characters')
        setIsLoading(false)
        return
      }

      // Simulate signup
      setTimeout(() => {
        setSuccess('Admin account created successfully! Logging in...')
        setTimeout(() => {
          login({
            username,
            isAdmin: true,
            loginTime: new Date().toISOString(),
          })
          navigate('/dashboard')
        }, 1000)
      }, 500)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-secondary via-gray-900 to-secondary flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-red-600 rounded-full mb-4">
            <Shield size={32} className="text-white" />
          </div>
          <h1 className="text-3xl font-bold text-white">Admin Panel</h1>
          <p className="text-gray-400 mt-2">
            {isLoginMode ? 'Administrator Login' : 'Create Admin Account'}
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="bg-gray-800 rounded-lg p-8 border border-gray-700 space-y-6">
            {/* Username */}
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-300 mb-2">
                Username
              </label>
              <input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={isLoading}
                placeholder="Enter admin username"
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-red-500 transition disabled:opacity-50"
              />
            </div>

            {/* Password */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isLoading}
                  placeholder="Enter password"
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-red-500 transition disabled:opacity-50"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-300 transition"
                  disabled={isLoading}
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
            </div>

            {/* Confirm Password (Signup only) */}
            {!isLoginMode && (
              <div>
                <label
                  htmlFor="confirmPassword"
                  className="block text-sm font-medium text-gray-300 mb-2"
                >
                  Confirm Password
                </label>
                <div className="relative">
                  <input
                    id="confirmPassword"
                    type={showPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    disabled={isLoading}
                    placeholder="Confirm password"
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-red-500 transition disabled:opacity-50"
                  />
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="bg-red-900 border border-red-700 text-red-200 px-4 py-3 rounded-lg text-sm">
                {error}
              </div>
            )}

            {/* Success Message */}
            {success && (
              <div className="bg-green-900 border border-green-700 text-green-200 px-4 py-3 rounded-lg text-sm">
                {success}
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-2 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white font-bold rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="loading-spinner" style={{ width: '20px', height: '20px' }}></div>
                  <span>{isLoginMode ? 'Logging in...' : 'Creating account...'}</span>
                </>
              ) : (
                <>
                  {isLoginMode ? <Shield size={20} /> : <UserPlus size={20} />}
                  <span>{isLoginMode ? 'Login as Admin' : 'Create Admin Account'}</span>
                </>
              )}
            </button>
          </div>

          {/* Mode Toggle */}
          <div className="text-center">
            <p className="text-gray-400">
              {isLoginMode ? 'New admin? ' : 'Already have an account? '}
              <button
                type="button"
                onClick={() => {
                  setIsLoginMode(!isLoginMode)
                  setError('')
                  setSuccess('')
                }}
                className="text-accent hover:underline font-semibold"
              >
                {isLoginMode ? 'Create Account' : 'Login'}
              </button>
            </p>
          </div>
        </form>

        {/* Links */}
        <div className="text-center mt-6">
          <p className="text-gray-400">
            Regular user? Go to{' '}
            <button
              onClick={() => navigate('/login')}
              className="text-accent hover:underline font-semibold"
            >
              User Login
            </button>
          </p>
        </div>

        {/* Security Notice */}
        <div className="mt-6 bg-yellow-900 border border-yellow-700 rounded-lg p-4">
          <p className="text-yellow-200 text-sm">
            <strong>Security Notice:</strong> Admin accounts have full control. Use strong
            passwords.
          </p>
        </div>
      </div>
    </div>
  )
}
