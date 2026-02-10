'use client'

import { useState } from 'react'

interface Technique {
  technique: string
  accuracy: number
  completeness: number
  efficiency: number
  overall: number
}

interface BenchmarkResult {
  best_result: {
    technique: string
    overall: number
    accuracy: number
    completeness: number
    efficiency: number
    response: string
    metrics: {
      latency: number
      total_tokens: number
    }
  }
  comparison: Technique[]
  all_responses: Record<string, { response: string; score: number }>
}

export default function Home() {
  const [problem, setProblem] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<BenchmarkResult | null>(null)
  const [error, setError] = useState('')

  // Get the best technique from comparison array (highest overall score)
  const getBestTechnique = () => {
    if (!result?.comparison || result.comparison.length === 0) return null
    
    return result.comparison.reduce((best, current) => {
      return (current.overall > best.overall) ? current : best
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch('http://localhost:8000/benchmark', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="h-screen overflow-hidden p-8 bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="max-w-7xl mx-auto h-full flex flex-col">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          Prompt Optimization Framework
        </h1>
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
                  htmlFor="problem"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                 Math Problem
                </label>
                <textarea
                  id="problem"
                  value={problem}
                  onChange={(e) => setProblem(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900"
                  rows={8}
                  placeholder="Enter your problem here (e.g., Solve for x: 2x + 5 = 15)"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
              >
                {loading ? 'Running Benchmark...' : 'Run Benchmark'}
              </button>
            </form>

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
                <p className="text-red-800 text-sm">
                  <strong>Error:</strong> {error}
                </p>
                <p className="text-red-600 text-xs mt-1">
                  Make sure the API server is running: uvicorn main:app --reload
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
                {/* Best Result */}
                <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                SOLUTION
              </h2>

              <div className="mt-4 p-4 bg-gray-50 rounded">
                <p className="text-sm text-gray-600 mb-2">
                  <strong>Response:</strong>
                </p>
                <p className="text-gray-900 whitespace-pre-wrap">
                  {result.best_result?.response || 'No response'}
                </p>
              </div>
            </div>

                {/* Scores */}
                <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                Best Technique: {getBestTechnique()?.technique?.toUpperCase() || 'N/A'}
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 p-4 rounded">
                  <p className="text-sm text-gray-600 mb-1">Overall Score</p>
                  <p className="text-3xl font-bold text-blue-600">
                    {getBestTechnique()?.overall?.toFixed(3) || '0.000'}
                  </p>
                </div>
                <div className="bg-green-50 p-4 rounded">
                  <p className="text-sm text-gray-600 mb-1">Accuracy</p>
                  <p className="text-3xl font-bold text-green-600">
                    {getBestTechnique()?.accuracy?.toFixed(3) || '0.000'}
                  </p>
                </div>
                <div className="bg-purple-50 p-4 rounded">
                  <p className="text-sm text-gray-600 mb-1">Completeness</p>
                  <p className="text-3xl font-bold text-purple-600">
                    {getBestTechnique()?.completeness?.toFixed(3) || '0.000'}
                  </p>
                </div>
                <div className="bg-orange-50 p-4 rounded">
                  <p className="text-sm text-gray-600 mb-1">Efficiency</p>
                  <p className="text-3xl font-bold text-orange-600">
                    {getBestTechnique()?.efficiency?.toFixed(3) || '0.000'}
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
                    </tr>
                  </thead>
                  <tbody>
                    {result.comparison?.map((tech) => {
                      const bestTech = getBestTechnique()
                      const isBest = tech.technique === bestTech?.technique
                      return (
                        <tr
                          key={tech.technique}
                          className={`border-b border-gray-100 ${
                            isBest ? 'bg-blue-50 font-semibold' : 'hover:bg-gray-50'
                          }`}
                        >
                          <td className="py-3 px-4 text-gray-900">
                            {tech.technique?.toUpperCase() || 'N/A'}
                            {isBest && ' '}
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
                        </tr>
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
