import { useNavigate } from 'react-router-dom'
import { Shield, Camera, AlertTriangle, Zap, BarChart3, Lock } from 'lucide-react'

export default function Home() {
  const navigate = useNavigate()

  const features = [
    {
      icon: Camera,
      title: 'Real-time Detection',
      description: 'YOLOv8-powered phone detection with 45%+ confidence threshold',
      color: 'from-blue-500 to-cyan-500',
    },
    {
      icon: AlertTriangle,
      title: 'Instant Alerts',
      description: 'Get alerts with grid coordinates when phones are detected',
      color: 'from-red-500 to-pink-500',
    },
    {
      icon: Lock,
      title: 'Secure Access',
      description: 'Role-based authentication with admin controls',
      color: 'from-purple-500 to-indigo-500',
    },
    {
      icon: BarChart3,
      title: 'Grid Mapping',
      description: '3x4 grid system for precise location identification',
      color: 'from-green-500 to-emerald-500',
    },
    {
      icon: Zap,
      title: 'Auto Recording',
      description: 'Automatic video recording for 5 seconds post-detection',
      color: 'from-yellow-500 to-orange-500',
    },
    {
      icon: Shield,
      title: 'Video Archive',
      description: 'Access historical records with full metadata',
      color: 'from-indigo-500 to-blue-500',
    },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-secondary via-gray-900 to-secondary">
      {/* Navigation */}
      <nav className="bg-secondary border-b border-gray-700 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 bg-accent rounded-lg flex items-center justify-center">
              <Shield size={24} className="text-secondary" />
            </div>
            <span className="text-xl font-bold text-accent">AI Surveillance</span>
          </div>
          <div className="flex gap-4">
            <button
              onClick={() => navigate('/login')}
              className="px-6 py-2 rounded-lg bg-transparent border border-gray-600 text-gray-300 hover:border-gray-400 hover:text-white transition"
            >
              Login
            </button>
            <button
              onClick={() => navigate('/admin-login')}
              className="px-6 py-2 rounded-lg bg-accent text-secondary font-bold hover:opacity-90 transition"
            >
              Admin
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid md:grid-cols-2 gap-12 items-center mb-20">
          <div>
            <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 leading-tight">
              Advanced <span className="text-accent">Phone Detection</span> System
            </h1>
            <p className="text-xl text-gray-400 mb-8">
              Powered by YOLOv8, monitor exam halls in real-time with intelligent phone detection,
              instant alerts, and complete video recording for security and compliance.
            </p>
            <div className="flex gap-4">
              <button
                onClick={() => navigate('/login')}
                className="px-8 py-3 bg-primary hover:bg-blue-700 text-white rounded-lg font-semibold transition"
              >
                Get Started
              </button>
              <button
                onClick={() => {
                  document.getElementById('features').scrollIntoView({ behavior: 'smooth' })
                }}
                className="px-8 py-3 border border-accent text-accent hover:bg-accent hover:text-secondary rounded-lg font-semibold transition"
              >
                Learn More
              </button>
            </div>
          </div>
          <div className="relative h-96">
            <div className="absolute inset-0 bg-gradient-to-br from-primary via-blue-600 to-cyan-600 rounded-2xl opacity-20 blur-3xl"></div>
            <div className="relative bg-gradient-to-br from-blue-900 to-cyan-900 rounded-2xl p-8 border border-blue-700 h-full flex flex-col justify-between">
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-green-400 text-sm font-semibold">LIVE MONITORING</span>
                </div>
                <div className="text-2xl font-mono text-cyan-400">
                  Grid Position: R2C3
                </div>
                <div className="text-sm text-gray-300">
                  Confidence: <span className="text-green-400 font-bold">95%</span>
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-xs text-gray-400">Detection Details</div>
                <div className="text-xs text-gray-300 space-y-1">
                  <p>• Center: (320, 240)</p>
                  <p>• BBox: 100x150</p>
                  <p>• Status: RECORDING</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <section id="features" className="py-20">
          <h2 className="text-4xl font-bold text-white text-center mb-16">Key Features</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, idx) => {
              const Icon = feature.icon
              return (
                <div
                  key={idx}
                  className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-8 border border-gray-700 hover:border-gray-600 transition group"
                >
                  <div
                    className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4 group-hover:scale-110 transition`}
                  >
                    <Icon size={24} className="text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-white mb-2">{feature.title}</h3>
                  <p className="text-gray-400">{feature.description}</p>
                </div>
              )
            })}
          </div>
        </section>

        {/* Tech Stack */}
        <section className="py-20 text-center">
          <h2 className="text-4xl font-bold text-white mb-12">Technology Stack</h2>
          <div className="grid md:grid-cols-4 gap-8">
            {[
              { name: 'YOLOv8', desc: 'AI Detection' },
              { name: 'Flask', desc: 'Backend API' },
              { name: 'React', desc: 'Frontend' },
              { name: 'OpenCV', desc: 'Video Processing' },
            ].map((tech, idx) => (
              <div key={idx} className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <p className="text-accent font-bold text-xl">{tech.name}</p>
                <p className="text-gray-400 text-sm">{tech.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* CTA */}
        <section className="py-20 text-center">
          <div className="bg-gradient-to-r from-primary to-cyan-600 rounded-2xl p-12 border border-cyan-500">
            <h2 className="text-3xl font-bold text-white mb-4">Ready to Secure Your Space?</h2>
            <p className="text-lg text-blue-100 mb-8">
              Login to start monitoring exam halls with AI-powered detection
            </p>
            <button
              onClick={() => navigate('/login')}
              className="px-10 py-3 bg-white text-primary rounded-lg font-bold hover:opacity-90 transition"
            >
              Login Now
            </button>
          </div>
        </section>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-700 bg-secondary py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-gray-400">
          <p>© 2026 AI Surveillance System. All rights reserved.</p>
          <p className="text-sm mt-2">Unauthorized access prohibited</p>
        </div>
      </footer>
    </div>
  )
}
