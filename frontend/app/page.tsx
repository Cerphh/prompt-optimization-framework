'use client'

import { useState, useEffect } from 'react'

interface Technique {
  technique: string
  accuracy: number
  completeness: number
  efficiency: number
  overall: number
}

interface TechniqueResult {
  technique: string
  success: boolean
  prompt: string
  response: string
  metrics: {
    elapsed_time: number
    total_tokens: number
    prompt_tokens: number
    completion_tokens: number
  }
  scores: {
    accuracy: number
    completeness: number
    efficiency: number
    overall: number
  }
}

interface BenchmarkResult {
  problem: string
  best_technique: string
  best_result: TechniqueResult
  all_results: Record<string, TechniqueResult>
  comparison: Technique[]
  all_responses: Record<string, { response: string; score: number }>
}

export default function Home() {
  const [problem, setProblem] = useState('')
  const [subject, setSubject] = useState('algebra')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<BenchmarkResult | null>(null)
  const [error, setError] = useState('')
  const [expandedTechnique, setExpandedTechnique] = useState<string | null>(null)
  const [healthStatus, setHealthStatus] = useState<'checking' | 'healthy' | 'unhealthy'>('checking')
  const [validationError, setValidationError] = useState('')

  // Check API and Ollama health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('http://localhost:8000/health', { timeout: 3000 } as any)
        if (response.ok) {
          const data = await response.json()
          setHealthStatus(data.model_connected ? 'healthy' : 'unhealthy')
        } else {
          setHealthStatus('unhealthy')
        }
      } catch {
        setHealthStatus('unhealthy')
      }
    }
    
    checkHealth()
    const interval = setInterval(checkHealth, 10000) // Check every 10s
    return () => clearInterval(interval)
  }, [])

  // Auto-detect subject based on keywords in the problem
  const detectSubject = (text: string): string => {
    const lowerText = text.toLowerCase()
    
    // Keywords for each subject (ordered by specificity)
    const calculusKeywords = ['derivative', 'integral', 'limit', 'differentiate', 'integrate', 
                              'tangent line', 'rate of change', 'optimization', 'concave', 'inflection']
    const statisticsKeywords = ['probability', 'mean', 'median', 'mode', 'variance', 'standard deviation',
                                'distribution', 'expected value', 'random', 'coin', 'dice', 'sample']
    const algebraKeywords = ['solve for', 'factor', 'simplify', 'expand', 'quadratic', 'equation',
                            'polynomial', 'exponential', 'logarithm', 'inequality']
    
    // Check calculus first (most specific)
    if (calculusKeywords.some(keyword => lowerText.includes(keyword))) {
      return 'calculus'
    }
    
    // Then statistics
    if (statisticsKeywords.some(keyword => lowerText.includes(keyword))) {
      return 'statistics'
    }
    
    // Then algebra
    if (algebraKeywords.some(keyword => lowerText.includes(keyword))) {
      return 'algebra'
    }
    
    // Default to algebra for general math problems
    return 'algebra'
  }

  const handleProblemChange = (text: string) => {
    setProblem(text)
    
    // Input validation
    if (text.length > 0 && text.length < 10) {
      setValidationError('Problem must be at least 10 characters')
    } else {
      setValidationError('')
    }
    
    // Auto-detect and update subject if text is long enough
    if (text.length > 5) {
      const detectedSubject = detectSubject(text)
      setSubject(detectedSubject)
    }
  }

  // Export results as JSON
  const exportResults = () => {
    if (!result) return
    const dataStr = JSON.stringify(result, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `benchmark-${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    // Validate before submitting
    if (problem.length < 10) {
      setValidationError('Problem must be at least 10 characters')
      return
    }
    
    setLoading(true)
    setError('')
    setResult(null)

    try {
      // Check connection first
      if (healthStatus === 'unhealthy') {
        throw new Error('⚠️ System offline: Make sure Ollama is running (ollama serve) and the backend API is started')
      }

      const response = await fetch('http://localhost:8000/benchmark', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem,
          subject,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        const detail = errorData.detail || 'Unknown error'
        
        if (response.status === 500) {
          throw new Error(`🔴 Server Error: ${detail}`)
        } else if (response.status === 404) {
          throw new Error('🔴 API endpoint not found. Is the backend running?')
        } else {
          throw new Error(`🔴 HTTP ${response.status}: ${detail}`)
        }
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError('🔴 Cannot connect to API. Make sure backend is running on port 8000')
      } else {
        setError(err instanceof Error ? err.message : 'An error occurred')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="h-screen overflow-hidden p-8 bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="max-w-7xl mx-auto h-full flex flex-col">
        <div className="flex items-center justify-between mb-2">
          <h1 className="text-4xl font-bold text-gray-900">
            Prompt Optimization Framework
          </h1>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${healthStatus === 'healthy' ? 'bg-green-500' : healthStatus === 'unhealthy' ? 'bg-red-500' : 'bg-yellow-500'}`}></div>
            <span className="text-sm text-gray-600">
              {healthStatus === 'healthy' ? '🟢 Online' : healthStatus === 'unhealthy' ? '🔴 Offline' : '🟡 Checking...'}
            </span>
          </div>
        </div>
        <p className="text-gray-600 mb-6">
          Compare different prompting techniques and find the optimal approach
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1 overflow-hidden">
          {/* Left Side - Input Form */}
          <div className="bg-white rounded-lg shadow-md p-6 h-fit">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Input</h2>
            <form onSubmit={handleSubmit}>
              <div className="mb-4">
                <label
                  htmlFor="subject"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Subject Category <span className="text-xs text-gray-500">(auto-detected)</span>
                </label>
                <select
                  id="subject"
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900"
                >
                  <option value="algebra">Algebra</option>
                  <option value="statistics">Statistics & Probability</option>
                  <option value="calculus">Calculus</option>
                </select>
              </div>

              <div className="mb-4">
                <label
                  htmlFor="problem"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                 Math Problem
                </label>
                <textarea
                  id="problem"
                  value={problem}
                  onChange={(e) => handleProblemChange(e.target.value)}
                  className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 text-gray-900 ${
                    validationError ? 'border-red-300 focus:ring-red-500' : 'border-gray-300 focus:ring-blue-500'
                  }`}
                  rows={8}
                  placeholder="Enter your problem here (e.g., Solve for x: 2x + 5 = 15)"
                  required
                />
                {validationError && (
                  <p className="mt-1 text-sm text-red-600">{validationError}</p>
                )}
              </div>

              <button
                type="submit"
                disabled={loading || !!validationError || problem.length < 10}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
              >
                {loading ? 'Running Benchmark...' : 'Run Benchmark'}
              </button>
            </form>

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
                <p className="text-red-800 text-sm whitespace-pre-wrap">
                  {error}
                </p>
              </div>
            )}
          </div>

          {/* Right Side - Results */}
          <div className="overflow-y-auto space-y-6">
            {!result && !loading && (
              <div className="bg-white rounded-lg shadow-md p-12 text-center">
                <div className="text-gray-400 mb-4">
                  <svg
                    className="w-24 h-24 mx-auto"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </div>
                <p className="text-gray-600 text-lg">
                  Enter a problem and click "Run Benchmark" to see results
                </p>
              </div>
            )}

            {loading && (
              <div className="bg-white rounded-lg shadow-md p-12 text-center">
                <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-gray-600 text-lg">Running benchmark...</p>
                <p className="text-gray-500 text-sm mt-2">
                  This may take a few minutes
                </p>
              </div>
            )}

                {result && (
              <>
                {/* Best Technique Prompt and Response */}
                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-2xl font-bold text-gray-900">
                      Best Technique: {result.best_technique?.toUpperCase() || 'N/A'}
                    </h2>
                    <button
                      onClick={exportResults}
                      className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      Export JSON
                    </button>
                  </div>
                  
                  {/* Prompt Used */}
                  <div className="mb-6">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-lg font-semibold text-gray-800">Prompt Used</h3>
                      <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                        {result.best_technique}
                      </span>
                    </div>
                    <div className="p-4 bg-gray-50 rounded border border-gray-200">
                      <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono">
                        {result.best_result?.prompt || 'No prompt available'}
                      </pre>
                    </div>
                  </div>

                  {/* Model Response */}
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-2">Model Response</h3>
                    <div className="p-4 bg-green-50 rounded border border-green-200">
                      <pre className="text-sm text-gray-900 whitespace-pre-wrap">
                        {result.best_result?.response || 'No response'}
                      </pre>
                    </div>
                  </div>
                </div>

                {/* Scores */}
                <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                Performance Scores
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 p-4 rounded">
                  <p className="text-sm text-gray-600 mb-1">Overall Score</p>
                  <p className="text-3xl font-bold text-blue-600">
                    {result.best_result?.scores?.overall?.toFixed(3) || '0.000'}
                  </p>
                </div>
                <div className="bg-green-50 p-4 rounded">
                  <p className="text-sm text-gray-600 mb-1">Accuracy</p>
                  <p className="text-3xl font-bold text-green-600">
                    {result.best_result?.scores?.accuracy?.toFixed(3) || '0.000'}
                  </p>
                </div>
                <div className="bg-purple-50 p-4 rounded">
                  <p className="text-sm text-gray-600 mb-1">Completeness</p>
                  <p className="text-3xl font-bold text-purple-600">
                    {result.best_result?.scores?.completeness?.toFixed(3) || '0.000'}
                  </p>
                </div>
                <div className="bg-orange-50 p-4 rounded">
                  <p className="text-sm text-gray-600 mb-1">Efficiency</p>
                  <p className="text-3xl font-bold text-orange-600">
                    {result.best_result?.scores?.efficiency?.toFixed(3) || '0.000'}
                  </p>
                </div>
              </div>
            </div>

                {/* Comparison Table */}
                <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                Technique Comparison
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b-2 border-gray-200">
                      <th className="text-left py-2 px-4 text-gray-700">
                        Technique
                      </th>
                      <th className="text-center py-2 px-4 text-gray-700">
                        Accuracy
                      </th>
                      <th className="text-center py-2 px-4 text-gray-700">
                        Completeness
                      </th>
                      <th className="text-center py-2 px-4 text-gray-700">
                        Efficiency
                      </th>
                      <th className="text-center py-2 px-4 text-gray-700">
                        Overall
                      </th>
                      <th className="text-center py-2 px-4 text-gray-700">
                        Details
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.comparison?.map((tech) => {
                      const isBest = tech.technique === result.best_technique
                      const techResult = result.all_results[tech.technique]
                      const isExpanded = expandedTechnique === tech.technique
                      return (
                        <>
                          <tr
                            key={tech.technique}
                            className={`border-b border-gray-100 ${
                              isBest ? 'bg-blue-50 font-semibold' : 'hover:bg-gray-50'
                            }`}
                          >
                            <td className="py-3 px-4 text-gray-900">
                              {tech.technique?.toUpperCase() || 'N/A'}
                              {isBest && ' ⭐'}
                            </td>
                            <td className="py-3 px-4 text-center text-gray-700">
                              {tech.accuracy?.toFixed(3) || '0.000'}
                            </td>
                            <td className="py-3 px-4 text-center text-gray-700">
                              {tech.completeness?.toFixed(3) || '0.000'}
                            </td>
                            <td className="py-3 px-4 text-center text-gray-700">
                              {tech.efficiency?.toFixed(3) || '0.000'}
                            </td>
                            <td className="py-3 px-4 text-center font-bold text-blue-600">
                              {tech.overall?.toFixed(3) || '0.000'}
                            </td>
                            <td className="py-3 px-4 text-center">
                              <button
                                onClick={() => setExpandedTechnique(isExpanded ? null : tech.technique)}
                                className="text-blue-600 hover:text-blue-800 text-sm underline"
                              >
                                {isExpanded ? 'Hide' : 'View'}
                              </button>
                            </td>
                          </tr>
                          {isExpanded && techResult && (
                            <tr className="bg-gray-50">
                              <td colSpan={6} className="p-4">
                                <div className="space-y-4">
                                  <div>
                                    <h4 className="font-semibold text-gray-800 mb-2">Prompt:</h4>
                                    <pre className="text-xs text-gray-700 whitespace-pre-wrap bg-white p-3 rounded border border-gray-200 font-mono">
                                      {techResult.prompt}
                                    </pre>
                                  </div>
                                  <div>
                                    <h4 className="font-semibold text-gray-800 mb-2">Response:</h4>
                                    <pre className="text-xs text-gray-900 whitespace-pre-wrap bg-white p-3 rounded border border-gray-200">
                                      {techResult.response}
                                    </pre>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
              </>
            )}
          </div>
        </div>
      </div>
    </main>
  )
}
