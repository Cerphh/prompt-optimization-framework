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
    done_reason?: string
    truncated?: boolean
    continuation_rounds?: number
    continuation_error?: string | null
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
  selection_source?: 'db_history' | 'runtime_scores'
  selection_details?: {
    total_samples?: number
  }
  storage?: {
    success: boolean
    document_id?: string
    error?: string
  }
}

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')

const apiUrl = (path: string): string => `${API_BASE_URL}${path}`

/* Score color mapping */
const scoreColor = (v: number): string => {
  if (v >= 0.95) return 'var(--green)'
  if (v >= 0.80) return 'var(--blue)'
  if (v >= 0.65) return 'var(--amber)'
  return 'var(--text)'
}

export default function Home() {
  const [problem, setProblem] = useState('')
  const [subject, setSubject] = useState('algebra')
  const [difficulty, setDifficulty] = useState('basic')
  const [difficultyManualOverride, setDifficultyManualOverride] = useState(false)
  const [showBenchmarkOptions, setShowBenchmarkOptions] = useState(false)
  const [groundTruth, setGroundTruth] = useState('')
  const [lastRunUsedGroundTruth, setLastRunUsedGroundTruth] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<BenchmarkResult | null>(null)
  const [error, setError] = useState('')
  const [expandedTechnique, setExpandedTechnique] = useState<string | null>(null)
  const [healthStatus, setHealthStatus] = useState<'checking' | 'healthy' | 'unhealthy'>('checking')
  const [validationError, setValidationError] = useState('')
  const [savingToDb, setSavingToDb] = useState(false)
  const [saveStatus, setSaveStatus] = useState('')
  const [didSaveToDb, setDidSaveToDb] = useState(false)
  const [didExportJson, setDidExportJson] = useState(false)
  const [streamingResponse, setStreamingResponse] = useState('')
  const [streamingTechnique, setStreamingTechnique] = useState('')
  const [streamingStatus, setStreamingStatus] = useState('')

  // Check API health
  useEffect(() => {
    const checkHealth = async () => {
      const controller = new AbortController()
      const timeoutId = window.setTimeout(() => controller.abort(), 4000)

      try {
        const response = await fetch(apiUrl('/health'), {
          signal: controller.signal,
        })

        if (!response.ok) {
          setHealthStatus('unhealthy')
          return
        }

        const health = await response.json().catch(() => null)
        // Mark as unhealthy when API is up but model connectivity is degraded.
        if (health?.model_connected === false) {
          setHealthStatus('unhealthy')
          return
        }

        setHealthStatus('healthy')
      } catch {
        setHealthStatus('unhealthy')
      } finally {
        clearTimeout(timeoutId)
      }
    }
    
    checkHealth()
    const interval = setInterval(checkHealth, 10000) // Check every 10s
    return () => clearInterval(interval)
  }, [])

  const normalizeDetectionText = (text: string): string => {
    const replacements: Record<string, string> = {
      '−': '-',
      '–': '-',
      '×': '*',
      '÷': '/',
      '⁰': '^0',
      '¹': '^1',
      '²': '^2',
      '³': '^3',
      '⁴': '^4',
      '⁵': '^5',
      '⁶': '^6',
      '⁷': '^7',
      '⁸': '^8',
      '⁹': '^9',
    }

    let normalized = text
    Object.entries(replacements).forEach(([source, target]) => {
      normalized = normalized.replaceAll(source, target)
    })

    return normalized.toLowerCase().replace(/\s+/g, ' ').trim()
  }

  const looksLikeAlgebraEquationProblem = (text: string): boolean => {
    const lowerText = normalizeDetectionText(text)

    if (!lowerText.includes('=')) return false

    const hasVariable = /\b[a-z]\b|\d+[a-z]|[a-z]\d/.test(lowerText)
    if (!hasVariable) return false

    const explicitPrecalculus = /\bd\/dx\b|\bdy\/dx\b|∫|\bderivative\b|\bdifferentiate\b|\bintegral\b|\bintegrate\b|\blimit\b|\blim\b|\bsin\b|\bcos\b|\btan\b|\bsec\b|\bcsc\b|\bcot\b|\barcsin\b|\barccos\b|\barctan\b/.test(lowerText)
    if (explicitPrecalculus) return false

    if (/\bsolve\b|\bsolution\b|\bsolutions\b|\broot\b|\broots\b|\bequation\b|\bpolynomial\b|\bfactor\b|\bfind x\b|\bsolve for\b/.test(lowerText)) {
      return true
    }

    if (/\b[a-z]\^?\d/.test(lowerText)) return true
    if (/[-+]?\d+[a-z]/.test(lowerText)) return true
    if (/\b[a-z]\s*[+\-*/]\s*[a-z0-9]/.test(lowerText)) return true

    return false
  }

  // Auto-detect subject based on keywords in the problem
  const detectSubject = (text: string): string => {
    const lowerText = normalizeDetectionText(text)

    // Early guard to prevent equation-solving prompts from being mislabeled as pre-calculus.
    if (looksLikeAlgebraEquationProblem(lowerText)) {
      return 'algebra'
    }

    let scores = { 'pre-calculus': 0, 'counting-probability': 0, algebra: 0 }
    
    // Comprehensive keyword lists for each subject
    const precalculusPatterns = {
      keywords: ['derivative', 'integral', 'limit', 'differentiate', 'integrate', 'tangent', 
                 'rate of change', 'optimization', 'concave', 'inflection', 'slope', 'curve',
                 'velocity', 'acceleration', 'extrema', 'maximum', 'minimum', 'gradient',
                 'second derivative', 'critical point', 'antiderivative', 'riemann', 'area under',
                 'function', 'exponential', 'logarithmic', 'sequence', 'series', 'convergence',
                 'graph', 'asymptote', 'domain', 'range', 'inverse function', 'composite function',
                 'trigonometric', 'sine', 'cosine', 'tangent function', 'periodic'],
      patterns: [/d\/dx/, /∫/, /lim|limit/, /derivative|derivatives/, /integral|integrate/i, 
             /exponential|logarithm|log\(/, /sin|cos|tan|graph|function/, /find.*(derivative|integral|limit|domain|range|asymptote)/],
      problemGoals: ['solve', 'find', 'graph', 'calculate', 'determine'],  // What user wants to do
      symbols: ['dx', 'dy', 'dy/dx', '∫', 'e^', 'ln(', 'sin', 'cos', 'tan', '^'],
      operations: ['derivative', 'integral', 'limit', 'exponential growth'],
      weight: 3
    }
    
    const countingProbabilityPatterns = {
      keywords: ['probability', 'mean', 'median', 'mode', 'variance', 'standard deviation',
                 'distribution', 'expected value', 'random', 'coin', 'dice', 'die', 'sample',
                 'permutation', 'combination', 'count', 'arrangement', 'selection',
                 'bell curve', 'normal distribution', 'quartile', 'percentile', 'z-score',
                 'frequency', 'event', 'outcome', 'counting', 'arrange', 'choose',
                 'factorial', 'nCr', 'nPr', 'odds', 'chance', 'flip', 'flipped', 'roll', 'rolled',
                 'ways', 'how many', 'different', 'possible outcomes', 'sample space',
                 'at least', 'probability of', 'likelihood', 'bayes', 'conditional'],
      patterns: [/probability|P\(/, /mean|average|median|mode/, /standard deviation|σ|variance/i,
                 /distribution/, /sample|population/, /permutation|combination|C\(|nCr|nPr/i,
                 /counting|arrange|flip|roll|die|dice|coin/, /how many ways|ways to/i, 
                 /in how many|possible.*outcomes|select|pick/i],
      problemGoals: ['count', 'probability', 'likelihood', 'chance', 'outcomes', 'arrangements'],  // What user wants
      symbols: ['μ', 'σ', 'P(', 'C(', 'n!', '!', '±'],
      operations: ['probability', 'counting', 'arrangement', 'permutation', 'combination'],
      weight: 2
    }
    
    const algebraPatterns = {
      keywords: ['solve', 'solve for', 'factor', 'factorize', 'simplify', 'expand', 'quadratic', 
                 'equation', 'polynomial', 'linear', 'matrix', 'system of equations', 'roots', 'zero', 'parabola',
                 'binomial', 'trinomial', 'monomial', 'rational', 'radical', 'algebraic', 'real solutions', 'real roots',
                 'expression', 'substitute', 'inequality', 'variable', 'coefficient'],
      patterns: [/solve (for|x|y|z)/, /factor|factorize/, /simplify/, /expand/, /quadratic/i,
                 /equation(s)?/, /polynomial/, /inequality/, /system.*equation/],
      problemGoals: ['solve', 'find', 'factor', 'simplify', 'expand'],
      symbols: ['=', '≠', '<', '>', '≤', '≥', 'x', 'x²', 'y'],
      operations: ['factor', 'simplify', 'solve', 'expand'],
      weight: 1
    }
    
    // Helper function to detect problem goal (what user wants to find)
    const detectProblemGoal = (text: string) => {
      const goals = {
        precalc: ['graph', 'sketch', 'solve for x', 'find x', 'find the derivative', 'find the integral', 
                  'find the limit', 'determine the function', 'calculate the slope'],
        counting: ['how many ways', 'in how many', 'probability', 'likelihood', 'chance', 'ways to arrange',
                   'how many different', 'possible outcomes', 'probability of']
      }
      
      let precalcGoals = goals.precalc.filter(g => lowerText.includes(g)).length
      let countingGoals = goals.counting.filter(g => lowerText.includes(g)).length
      
      return { precalcGoals, countingGoals }
    }
    
    // Check patterns for each subject
    const checkPatterns = (patterns: typeof precalculusPatterns) => {
      let score = 0
      
      // Check keywords
      if (patterns.keywords.some(kw => lowerText.includes(kw))) {
        score += patterns.weight
      }
      
      // Check regex patterns (more specific, higher weight)
      if (patterns.patterns.some(pat => pat.test(lowerText))) {
        score += patterns.weight * 1.5
      }
      
      // Check for common operations mentioned
      if (patterns.operations.some(op => lowerText.includes(op))) {
        score += patterns.weight * 1.2
      }
      
      return score
    }
    
    // Calculate base scores
    scores['pre-calculus'] = checkPatterns(precalculusPatterns)
    scores['counting-probability'] = checkPatterns(countingProbabilityPatterns)
    scores.algebra = checkPatterns(algebraPatterns)
    
    // Apply goal-based detection (helps distinguish pre-calc vs counting)
    const { precalcGoals, countingGoals } = detectProblemGoal(text)
    if (precalcGoals > countingGoals) {
      scores['pre-calculus'] += 2
    } else if (countingGoals > precalcGoals) {
      scores['counting-probability'] += 2
    }
    
    // Return subject with highest score, or algebra as default
    const maxScore = Math.max(scores['pre-calculus'], scores['counting-probability'], scores.algebra)
    if (maxScore === 0) return 'algebra' // Default if no patterns match
    
    if (scores['pre-calculus'] === maxScore) return 'pre-calculus'
    if (scores['counting-probability'] === maxScore) return 'counting-probability'
    return 'algebra'
  }

  const detectDifficulty = (text: string, detectedSubject: string): string => {
    const lowerText = text.toLowerCase()
    let score = 0

    // Subject-based scoring
    const subjectComplexity = {
      'pre-calculus': 3,
      'counting-probability': 1,
      'algebra': 0
    }
    
    score += subjectComplexity[detectedSubject as keyof typeof subjectComplexity] || 0

    // Advanced pre-calculus indicators
    const advancedPrecalculusKeywords = [
      'differentiate', 'derivative', 'partial derivative', 'integrate', 'integral', 
      'limit', 'optimization', 'inflection', 'concavity', 'series',
      'taylor', 'maclaurin', 'convergence', 'divergence', 'chain rule',
      'logarithmic differentiation', 'implicit differentiation', 'riemann sum'
    ]
    
    // Advanced counting & probability indicators
    const advancedCountingKeywords = [
      'bayes', 'conditional probability', 'given that', 'dependent', 'independent',
      'multivariate', 'distribution', 'binomial', 'poisson', 'hypergeometric',
      'permutation', 'combination', 'multinomial', 'expected value', 'variance'
    ]
    
    // Advanced algebra indicators
    const advancedAlgebraKeywords = [
      'system of equations', 'quadratic formula', 'imaginary', 'complex', 'matrix',
      'determinant', 'eigenvalue', 'eigenvector', 'rational function', 'partial fraction',
      'asymptote', 'domain', 'range', 'piecewise'
    ]
    
    // Intermediate indicators (common in multiple subjects)
    const intermediateKeywords = [
      'quadratic', 'factor', 'inequality', 'probability', 'permutation', 'combination',
      'standard deviation', 'variance', 'word problem', 'application', 'exponential',
      'logarithmic', 'solve the system', 'arrangement', 'selection'
    ]
    
    // Basic indicators (simple operations)
    const basicKeywords = [
      'add', 'subtract', 'multiply', 'divide', 'plus', 'minus', 'simplify',
      'evaluate', 'calculate', 'find', 'what is', 'compute', 'solve'
    ]

    // Check for difficulty keywords
    if (advancedPrecalculusKeywords.some(kw => lowerText.includes(kw))) score += 3
    else if (advancedCountingKeywords.some(kw => lowerText.includes(kw))) score += 3
    else if (advancedAlgebraKeywords.some(kw => lowerText.includes(kw))) score += 3
    
    if (intermediateKeywords.some(kw => lowerText.includes(kw))) score += 2
    if (basicKeywords.some(kw => lowerText.includes(kw))) score += 0.5

    // Structural complexity scoring
    const operatorCount = (text.match(/[+\-*/^=<>≤≥]/g) || []).length
    const parenthesesCount = (text.match(/[\(\)\[\]\{}]/g) || []).length
    const fractionCount = (text.match(/\//g) || []).length
    
    if (operatorCount >= 8) score += 2
    else if (operatorCount >= 5) score += 1.5
    else if (operatorCount >= 3) score += 1

    if (parenthesesCount >= 4) score += 1.5
    else if (parenthesesCount >= 2) score += 0.5

    if (fractionCount >= 3) score += 1
    else if (fractionCount >= 1) score += 0.5

    // Number complexity
    const numberCount = (text.match(/\d+/g) || []).length
    const largeNumbers = text.match(/\d{3,}/g) || []
    
    if (largeNumbers.length >= 2) score += 1
    if (numberCount >= 8) score += 1
    else if (numberCount >= 5) score += 0.5

    // Text length and complexity
    const wordCount = text.trim().split(/\s+/).filter(Boolean).length
    if (wordCount >= 40) score += 1
    else if (wordCount >= 25) score += 0.5

    // Multiple sentences or complex structure
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0).length
    if (sentences >= 3) score += 1

    // Determine difficulty tier
    if (score >= 6) return 'advanced'
    if (score >= 3) return 'intermediate'
    return 'basic'
  }

  const handleProblemChange = (text: string) => {
    setProblem(text)

    if (!text.trim()) {
      setSubject('algebra')
      setDifficultyManualOverride(false)
      if (!difficultyManualOverride) {
        setDifficulty('basic')
      }
    }
    
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

      if (!difficultyManualOverride) {
        const detectedDifficulty = detectDifficulty(text, detectedSubject)
        setDifficulty(detectedDifficulty)
      }
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
    setDidExportJson(true)
    if (didSaveToDb) {
      setSaveStatus('Exported as JSON and saved to DB.')
    } else {
      setSaveStatus('Exported as JSON. Don\u2019t forget to Save to DB as well.')
    }
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
    setSaveStatus('')
    setDidSaveToDb(false)
    setDidExportJson(false)
    setResult(null)
    setStreamingResponse('')
    setStreamingTechnique('')
    setStreamingStatus('Initializing benchmark...')

    try {
      // Check connection first
      if (healthStatus === 'unhealthy') {
        throw new Error('⚠️ System offline: Make sure Ollama is running (ollama serve) and the backend API is started')
      }

      const groundTruthValue = groundTruth.trim()
      setLastRunUsedGroundTruth(Boolean(groundTruthValue))

      const response = await fetch(apiUrl('/benchmark/stream'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem,
          subject,
          difficulty,
          speed_profile: 'fast',
          ...(groundTruthValue ? { ground_truth: groundTruthValue } : {}),
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

      if (!response.body) {
        throw new Error('Streaming not supported by this browser/session')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finalResult: BenchmarkResult | null = null
      let streamedText = ''
      let lastStreamingUiUpdate = 0

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.trim()) continue

          let event: any
          try {
            event = JSON.parse(line)
          } catch {
            continue
          }

          if (event.type === 'status') {
            if (event.technique) {
              setStreamingTechnique(event.technique)
            }
            setStreamingStatus(event.message || 'Running benchmark...')
          } else if (event.type === 'token') {
            streamedText += event.content || ''
            const now = Date.now()
            if (now - lastStreamingUiUpdate >= 500) {
              setStreamingResponse(streamedText)
              lastStreamingUiUpdate = now
            }
          } else if (event.type === 'complete') {
            finalResult = event.result
          } else if (event.type === 'error') {
            throw new Error(`🔴 Server Error: ${event.error || 'Unknown streaming error'}`)
          }
        }
      }

      if (streamedText) {
        setStreamingResponse(streamedText)
      }

      if (!finalResult) {
        throw new Error('Benchmark stream ended before returning final result')
      }

      setResult(finalResult)
      setSaveStatus('Not saved yet. Click Save to DB to save this result.')
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

  const handleSaveToDb = async () => {
    if (!result) return

    setSavingToDb(true)
    setSaveStatus('')

    try {
      const response = await fetch(apiUrl('/results/save'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          result,
          source: 'frontend_manual_save',
          metadata: {
            subject,
            difficulty,
          },
        }),
      })

      const data = await response.json().catch(() => ({}))

      if (!response.ok || !data?.storage) {
        throw new Error(data?.detail || data?.storage?.error || 'Failed to save result to DB')
      }

      setResult((prev) => {
        if (!prev) return prev
        return {
          ...prev,
          storage: data.storage,
        }
      })

      if (data.storage.success) {
        setDidSaveToDb(true)
        if (didExportJson) {
          setSaveStatus('Exported as JSON and saved to DB.')
        } else {
          setSaveStatus('Saved to DB successfully.')
        }
      } else {
        setSaveStatus(`Save failed: ${data.storage.error || 'Unknown error'}`)
      }
    } catch (err) {
      setSaveStatus(err instanceof Error ? err.message : 'Failed to save result to DB')
    } finally {
      setSavingToDb(false)
    }
  }

  /* Derived data for the technique-details modal */
  const expandedResult = expandedTechnique ? result?.all_results[expandedTechnique] : null
  const bestMetrics = result?.best_result?.metrics
  const bestContinuationRounds = bestMetrics?.continuation_rounds ?? 0
  const bestWasExtended = bestContinuationRounds > 0
  const bestStillTruncated = bestMetrics?.truncated === true

  const expandedMetrics = expandedResult?.metrics
  const expandedContinuationRounds = expandedMetrics?.continuation_rounds ?? 0
  const expandedWasExtended = expandedContinuationRounds > 0
  const expandedStillTruncated = expandedMetrics?.truncated === true

  const isDisabled = loading || !!validationError || problem.length < 10

  /* Whether we're in "results" mode (sidebar + right panel) */
  const hasStarted = loading || !!result

  return (
    <div className="h-screen flex flex-col" style={{ background: 'var(--bg)', color: 'var(--text)' }}>
      {/* ═══ Header ═══ */}
      <header
        className="flex items-center justify-between px-8 h-14 shrink-0"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        <h1 className="text-base font-semibold tracking-tight">Prompt Optimization Framework</h1>
        <div className="flex items-center gap-4">
          {result && (
            <span className="text-sm font-mono" style={{ color: 'var(--text-muted)' }}>
              Best:{' '}
              <strong className="font-medium" style={{ color: 'var(--text)' }}>
                {result.best_technique?.toUpperCase()}
              </strong>
              {' \u00b7 '}
              <strong className="font-medium" style={{ color: 'var(--text)' }}>
                {result.best_result?.scores?.overall?.toFixed(3)}
              </strong>
            </span>
          )}
          <span
            className="inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-0.5 rounded-full"
            style={{
              border: `1px solid ${
                healthStatus === 'healthy'
                  ? 'var(--green)'
                  : healthStatus === 'unhealthy'
                  ? '#ef4444'
                  : 'var(--amber)'
              }`,
              color:
                healthStatus === 'healthy'
                  ? 'var(--green)'
                  : healthStatus === 'unhealthy'
                  ? '#ef4444'
                  : 'var(--amber)',
            }}
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background:
                  healthStatus === 'healthy'
                    ? 'var(--green)'
                    : healthStatus === 'unhealthy'
                    ? '#ef4444'
                    : 'var(--amber)',
              }}
            />
            {healthStatus === 'healthy' ? 'online' : healthStatus === 'unhealthy' ? 'offline' : 'checking\u2026'}
          </span>
        </div>
      </header>

      {/* ═══ LANDING (before run) ═══ */}
      {!hasStarted && (
        <div className="flex-1 flex flex-col items-center justify-center px-4">
          <h2 className="text-3xl font-semibold mb-2 text-center">What would you like to benchmark?</h2>
          <p className="text-sm mb-8 text-center" style={{ color: 'var(--text-muted)' }}>
            Enter a problem below and compare how different prompting techniques perform.
          </p>

          <div
            className="w-full max-w-[920px] rounded-xl p-6"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
          >
            <form onSubmit={handleSubmit}>
              {/* Top row: Subject + Difficulty */}
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label htmlFor="subject-landing" className="flex items-center gap-2 text-sm font-medium mb-1.5">
                    Subject Category
                    <span
                      className="text-[10px] font-mono leading-none px-1.5 py-[3px] rounded"
                      style={{ color: 'var(--text-subtle)', border: '1px solid var(--border)' }}
                    >
                      auto-detected
                    </span>
                  </label>
                  <select
                    id="subject-landing"
                    value={subject}
                    onChange={(e) => setSubject(e.target.value)}
                    className="w-full px-3 py-2 rounded-md text-sm outline-none"
                    style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--surface)' }}
                  >
                    <option value="algebra">Algebra</option>
                    <option value="counting-probability">Counting &amp; Probability</option>
                    <option value="pre-calculus">Pre-calculus</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="difficulty-landing" className="flex items-center gap-2 text-sm font-medium mb-1.5">
                    Difficulty
                    <span
                      className="text-[10px] font-mono leading-none px-1.5 py-[3px] rounded"
                      style={{ color: 'var(--text-subtle)', border: '1px solid var(--border)' }}
                    >
                      override allowed
                    </span>
                  </label>
                  <select
                    id="difficulty-landing"
                    value={difficulty}
                    onChange={(e) => { setDifficulty(e.target.value); setDifficultyManualOverride(true) }}
                    className="w-full px-3 py-2 rounded-md text-sm outline-none"
                    style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--surface)' }}
                  >
                    <option value="basic">Basic</option>
                    <option value="intermediate">Intermediate</option>
                    <option value="advanced">Advanced</option>
                  </select>
                </div>
              </div>

              {/* Advanced benchmark options */}
              <div className="mb-4">
                <button
                  type="button"
                  onClick={() => setShowBenchmarkOptions((value) => !value)}
                  className="text-xs font-mono font-medium hover:underline"
                  style={{ color: 'var(--blue)' }}
                >
                  {showBenchmarkOptions ? 'Hide' : 'Show'} advanced benchmark options
                </button>

                {showBenchmarkOptions && (
                  <div
                    className="mt-2 p-3 rounded-md"
                    style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
                  >
                    <label htmlFor="ground-truth-landing" className="block text-sm font-medium mb-1.5">
                      Expected Answer (optional)
                    </label>
                    <input
                      id="ground-truth-landing"
                      value={groundTruth}
                      onChange={(e) => setGroundTruth(e.target.value)}
                      placeholder="e.g., x = 4 or x = 5"
                      className="w-full px-3 py-2 rounded-md text-sm font-mono outline-none"
                      style={{
                        border: '1px solid var(--border)',
                        color: 'var(--text)',
                        background: 'var(--surface)',
                      }}
                    />
                    <p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                      Leave blank for normal user solving. Fill this in for stronger benchmark accuracy scoring.
                    </p>
                  </div>
                )}
              </div>

              {/* Textarea */}
              <textarea
                value={problem}
                onChange={(e) => handleProblemChange(e.target.value)}
                rows={5}
                placeholder="Enter your problem here (e.g., Solve for x: 2x + 5 = 15)"
                className="w-full px-3 py-2 rounded-md text-sm font-mono outline-none resize-y"
                style={{
                  border: `1px solid ${validationError ? '#ef4444' : 'var(--border)'}`,
                  color: 'var(--text)',
                  background: 'var(--surface)',
                }}
                required
              />
              {validationError && (
                <p className="mt-1 text-xs" style={{ color: '#ef4444' }}>{validationError}</p>
              )}

              {/* Footer: techniques hint + button */}
              <div className="flex items-center justify-between mt-4">
                <span className="text-xs font-mono" style={{ color: 'var(--text-subtle)' }}>
                  Techniques: FEW_SHOT &middot; ZERO_SHOT
                </span>
                <button
                  type="submit"
                  disabled={isDisabled}
                  className="px-6 py-2.5 rounded-md text-sm font-medium transition-colors"
                  style={{
                    background: isDisabled ? 'var(--border-strong)' : 'var(--accent)',
                    color: isDisabled ? 'var(--text-muted)' : '#fff',
                    cursor: isDisabled ? 'not-allowed' : 'pointer',
                  }}
                >
                  Run Benchmark &rarr;
                </button>
              </div>
            </form>
          </div>

          {error && (
            <div
              className="mt-5 w-full max-w-[920px] p-3 rounded-md text-xs leading-relaxed whitespace-pre-wrap"
              style={{ background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b' }}
            >
              {error}
            </div>
          )}
        </div>
      )}

      {/* ═══ RESULTS MODE (sidebar + main) ═══ */}
      {hasStarted && (
        <div className="flex flex-1 overflow-hidden">
          {/* ─── Sidebar ─── */}
          <aside
            className="w-[320px] shrink-0 overflow-y-auto p-6"
            style={{ borderRight: '1px solid var(--border)' }}
          >
            <form onSubmit={handleSubmit} className="space-y-5">
              {/* Subject */}
              <div>
                <label htmlFor="subject" className="flex items-center gap-2 text-sm font-medium mb-1.5">
                  Subject Category
                  <span
                    className="text-[10px] font-mono leading-none px-1.5 py-[3px] rounded"
                    style={{ color: 'var(--text-subtle)', border: '1px solid var(--border)' }}
                  >
                    auto-detected
                  </span>
                </label>
                <select
                  id="subject"
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  disabled={loading}
                  className="w-full px-3 py-2 rounded-md text-sm outline-none transition disabled:cursor-not-allowed"
                  style={{
                    border: '1px solid var(--border)',
                    color: 'var(--text)',
                    background: loading ? 'var(--bg)' : 'var(--surface)',
                  }}
                >
                  <option value="algebra">Algebra</option>
                  <option value="counting-probability">Counting &amp; Probability</option>
                  <option value="pre-calculus">Pre-calculus</option>
                </select>
              </div>

              {/* Difficulty */}
              <div>
                <label htmlFor="difficulty" className="flex items-center gap-2 text-sm font-medium mb-1.5">
                  Difficulty Level
                  <span
                    className="text-[10px] font-mono leading-none px-1.5 py-[3px] rounded"
                    style={{ color: 'var(--text-subtle)', border: '1px solid var(--border)' }}
                  >
                    override allowed
                  </span>
                </label>
                <select
                  id="difficulty"
                  value={difficulty}
                  onChange={(e) => {
                    setDifficulty(e.target.value)
                    setDifficultyManualOverride(true)
                  }}
                  disabled={loading}
                  className="w-full px-3 py-2 rounded-md text-sm outline-none transition disabled:cursor-not-allowed"
                  style={{
                    border: '1px solid var(--border)',
                    color: 'var(--text)',
                    background: loading ? 'var(--bg)' : 'var(--surface)',
                  }}
                >
                  <option value="basic">Basic</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                </select>
              </div>

              {/* Advanced benchmark options */}
              <div>
                <button
                  type="button"
                  onClick={() => setShowBenchmarkOptions((value) => !value)}
                  className="text-xs font-mono font-medium hover:underline"
                  style={{ color: 'var(--blue)' }}
                >
                  {showBenchmarkOptions ? 'Hide' : 'Show'} advanced benchmark options
                </button>

                {showBenchmarkOptions && (
                  <div
                    className="mt-2 p-3 rounded-md"
                    style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
                  >
                    <label htmlFor="ground-truth" className="block text-sm font-medium mb-1.5">
                      Expected Answer (optional)
                    </label>
                    <input
                      id="ground-truth"
                      value={groundTruth}
                      onChange={(e) => setGroundTruth(e.target.value)}
                      disabled={loading}
                      placeholder="e.g., x = 4 or x = 5"
                      className="w-full px-3 py-2 rounded-md text-sm font-mono outline-none transition"
                      style={{
                        border: '1px solid var(--border)',
                        color: 'var(--text)',
                        background: loading ? 'var(--bg)' : 'var(--surface)',
                      }}
                    />
                    <p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                      Leave blank for normal user solving. Fill this in for stronger benchmark accuracy scoring.
                    </p>
                  </div>
                )}
              </div>

              {/* Problem */}
              <div>
                <label htmlFor="problem" className="block text-sm font-medium mb-1.5">
                  Math Problem
                </label>
                <textarea
                  id="problem"
                  value={problem}
                  onChange={(e) => handleProblemChange(e.target.value)}
                  disabled={loading}
                  rows={6}
                  placeholder="Enter your problem here (e.g., Solve for x: 2x + 5 = 15)"
                  className="w-full px-3 py-2 rounded-md text-sm font-mono outline-none resize-y transition"
                  style={{
                    border: `1px solid ${validationError ? '#ef4444' : 'var(--border)'}`,
                    color: 'var(--text)',
                    background: 'var(--surface)',
                  }}
                  required
                />
                {validationError && (
                  <p className="mt-1 text-xs" style={{ color: '#ef4444' }}>
                    {validationError}
                  </p>
                )}
              </div>

              {/* Submit */}
              <button
                type="submit"
                disabled={isDisabled}
                className="w-full py-2.5 rounded-md text-sm font-medium transition-colors"
                style={{
                  background: isDisabled ? 'var(--border-strong)' : 'var(--accent)',
                  color: isDisabled ? 'var(--text-muted)' : '#fff',
                  cursor: isDisabled ? 'not-allowed' : 'pointer',
                }}
              >
                {loading ? 'Running Benchmark\u2026' : 'Run Benchmark'}
              </button>
            </form>

            {error && (
              <div
                className="mt-5 p-3 rounded-md text-xs leading-relaxed whitespace-pre-wrap"
                style={{ background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b' }}
              >
                {error}
              </div>
            )}
          </aside>

          {/* ─── Main content ─── */}
          <section className="flex-1 overflow-y-auto p-8 space-y-6">
          {loading && (
            <div
              className="rounded-lg p-10 text-center"
              style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
            >
              <div
                className="animate-spin rounded-full h-10 w-10 mx-auto mb-4"
                style={{ border: '2px solid var(--border)', borderTopColor: 'var(--text)' }}
              />
              <p className="text-sm font-medium">{streamingStatus || 'Running benchmark\u2026'}</p>
              <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                This may take a few seconds
              </p>
              {streamingResponse && (
                <div className="mt-6 text-left">
                  <div className="flex items-center justify-between mb-2">
                    <span
                      className="text-[11px] font-mono uppercase tracking-wider"
                      style={{ color: 'var(--text-subtle)' }}
                    >
                      Live Output
                    </span>
                    <span
                      className="font-mono text-[11px] px-2 py-0.5 rounded"
                      style={{ color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                    >
                      {streamingTechnique || 'preview'}
                    </span>
                  </div>
                  <div
                    className="p-4 rounded-md max-h-64 overflow-y-auto"
                    style={{ background: '#f0fdf4', border: '1px solid #bbf7d0' }}
                  >
                    <pre className="text-sm font-mono whitespace-pre-wrap" style={{ color: 'var(--text)' }}>
                      {streamingResponse}
                      <span className="animate-pulse">{'\u258d'}</span>
                    </pre>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ═══ Results ═══ */}
          {result && (
            <>
              {/* ── Best Technique ── */}
              <div
                className="rounded-lg overflow-hidden"
                style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
              >
                {/* Header */}
                <div
                  className="flex items-center justify-between px-6 py-4"
                  style={{ borderBottom: '1px solid var(--border)' }}
                >
                  <div className="flex items-center gap-3">
                    <h2 className="text-lg font-semibold">Best Technique</h2>
                    <span
                      className="font-mono text-xs px-2.5 py-1 rounded"
                      style={{ background: 'var(--accent)', color: '#fff' }}
                    >
                      {result.best_technique?.toUpperCase()}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleSaveToDb}
                      disabled={savingToDb}
                      className="px-3 py-1.5 rounded-md text-xs font-medium transition-colors disabled:cursor-not-allowed"
                      style={{
                        border: '1px solid var(--border)',
                        color: 'var(--text)',
                        background: 'var(--surface)',
                      }}
                    >
                      {savingToDb ? 'Saving\u2026' : 'Save to DB'}
                    </button>
                    <button
                      onClick={exportResults}
                      className="px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
                      style={{ background: 'var(--accent)', color: '#fff' }}
                    >
                      Export JSON
                    </button>
                  </div>
                </div>

                {/* Save / Export status */}
                {saveStatus && (() => {
                  const isSuccess = saveStatus.startsWith('Saved') || saveStatus.startsWith('Exported as JSON and saved')
                  const isExportOnly = saveStatus.startsWith('Exported as JSON. Don')
                  const isNotSaved = saveStatus.startsWith('Not saved')
                  const isFail = !isSuccess && !isExportOnly && !isNotSaved

                  let bg = '#fffbeb'
                  let borderColor = '#fde68a'
                  let textColor = 'var(--amber)'
                  let icon = '\u26a0 '

                  if (isSuccess) {
                    bg = '#f0fdf4'; borderColor = '#bbf7d0'; textColor = 'var(--green)'; icon = '\u2713 '
                  } else if (isExportOnly) {
                    bg = '#eff6ff'; borderColor = '#bfdbfe'; textColor = 'var(--blue)'; icon = '\u2193 '
                  } else if (isFail) {
                    bg = '#fef2f2'; borderColor = '#fecaca'; textColor = '#dc2626'; icon = '\u2715 '
                  }

                  return (
                    <div
                      className="px-6 py-2 text-xs"
                      style={{ background: bg, borderBottom: `1px solid ${borderColor}`, color: textColor }}
                    >
                      {icon}
                      {isExportOnly ? (
                        <>Exported as JSON. Don{'\u2019'}t forget to <strong>Save to DB</strong> as well.</>
                      ) : isNotSaved ? (
                        <>Not saved yet. Click <strong>Save to DB</strong> to save this result.</>
                      ) : (
                        saveStatus
                      )}
                    </div>
                  )
                })()}

                {(bestWasExtended || bestStillTruncated) && (
                  <div
                    className="px-6 py-2 text-xs"
                    style={{
                      background: bestStillTruncated ? '#fef2f2' : '#eff6ff',
                      borderBottom: `1px solid ${bestStillTruncated ? '#fecaca' : '#bfdbfe'}`,
                      color: bestStillTruncated ? '#dc2626' : 'var(--blue)',
                    }}
                  >
                    {bestStillTruncated
                      ? '⚠ Response may still be incomplete due model token limit. Try increasing MODEL_NUM_PREDICT or MODEL_MAX_CONTINUE_ROUNDS.'
                      : `↻ Long-answer safeguard used: ${bestContinuationRounds} continuation round${bestContinuationRounds === 1 ? '' : 's'} to finish the response.`}
                  </div>
                )}

                {/* Body */}
                <div className="p-6 space-y-6">
                  {/* Prompt Used */}
                  <div>
                    <p
                      className="text-[11px] font-mono uppercase tracking-wider mb-2"
                      style={{ color: 'var(--text-subtle)' }}
                    >
                      Prompt Used
                    </p>
                    <div
                      className="p-4 rounded-md overflow-auto max-h-72"
                      style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
                    >
                      <pre className="text-sm font-mono whitespace-pre-wrap leading-relaxed">
                        {result.best_result?.prompt || 'No prompt available'}
                      </pre>
                    </div>
                  </div>

                  {/* Model Response */}
                  <div>
                    <p
                      className="text-[11px] font-mono uppercase tracking-wider mb-2"
                      style={{ color: 'var(--text-subtle)' }}
                    >
                      Model Response
                    </p>
                    <div
                      className="p-4 rounded-md overflow-auto max-h-96"
                      style={{ background: '#f0fdf4', border: '1px solid #bbf7d0' }}
                    >
                      <pre className="text-sm font-mono whitespace-pre-wrap leading-relaxed">
                        {result.best_result?.response || 'No response'}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>

              {/* ── Performance Scores ── */}
              <div
                className="rounded-lg p-6"
                style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
              >
                <h2 className="text-lg font-semibold mb-4">Performance Scores</h2>

                {!lastRunUsedGroundTruth && (
                  <div
                    className="mb-4 px-3 py-2 rounded-md text-xs"
                    style={{ background: '#fffbeb', border: '1px solid #fde68a', color: 'var(--amber)' }}
                  >
                    Accuracy is heuristic for this run because no expected answer was provided.
                  </div>
                )}

                <div className="grid grid-cols-4 gap-4">
                  {[
                    { label: 'OVERALL', value: result.best_result?.scores?.overall },
                    {
                      label: !lastRunUsedGroundTruth ? 'ACCURACY*' : 'ACCURACY',
                      value: result.best_result?.scores?.accuracy,
                    },
                    { label: 'COMPLETENESS', value: result.best_result?.scores?.completeness },
                    { label: 'EFFICIENCY', value: result.best_result?.scores?.efficiency },
                  ].map((s) => (
                    <div
                      key={s.label}
                      className="p-4 rounded-md text-center"
                      style={{ border: '1px solid var(--border)' }}
                    >
                      <p
                        className="text-[10px] font-mono uppercase tracking-widest mb-2"
                        style={{ color: 'var(--text-subtle)' }}
                      >
                        {s.label}
                      </p>
                      <p
                        className="text-3xl font-mono font-light"
                        style={{ color: scoreColor(s.value ?? 0) }}
                      >
                        {s.value?.toFixed(3) ?? '0.000'}
                      </p>
                    </div>
                  ))}
                </div>

                {!lastRunUsedGroundTruth && (
                  <p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                    * Heuristic accuracy can be directionally useful, but is less reliable than ground-truth scoring.
                  </p>
                )}
              </div>

              {/* ── Technique Comparison ── */}
              <div
                className="rounded-lg overflow-hidden"
                style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
              >
                <div
                  className="flex items-center justify-between px-6 py-4"
                  style={{ borderBottom: '1px solid var(--border)' }}
                >
                  <h2 className="text-lg font-semibold">Technique Comparison</h2>
                  <span className="font-mono text-xs" style={{ color: 'var(--text-subtle)' }}>
                    {result.comparison?.length ?? 0} techniques
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border)' }}>
                        {['Technique', 'Accuracy', 'Completeness', 'Efficiency', 'Overall', 'Details'].map(
                          (col, i) => (
                            <th
                              key={col}
                              className={`${
                                i === 0 ? 'text-left px-6' : 'text-center px-4'
                              } py-3 text-[10px] font-mono uppercase tracking-widest font-medium`}
                              style={{ color: 'var(--text-subtle)' }}
                            >
                              {col}
                            </th>
                          )
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {result.comparison?.map((tech) => {
                        const isBest = tech.technique === result.best_technique
                        return (
                          <tr
                            key={tech.technique}
                            className={isBest ? 'font-medium' : ''}
                            style={{ borderBottom: '1px solid var(--border)' }}
                          >
                            <td className="px-6 py-3 font-medium">
                              {tech.technique?.toUpperCase() ?? 'N/A'}
                            </td>
                            <td className="text-center px-4 py-3 font-mono">
                              {tech.accuracy?.toFixed(3) ?? '0.000'}
                            </td>
                            <td className="text-center px-4 py-3 font-mono">
                              {tech.completeness?.toFixed(3) ?? '0.000'}
                            </td>
                            <td className="text-center px-4 py-3 font-mono">
                              {tech.efficiency?.toFixed(3) ?? '0.000'}
                            </td>
                            <td
                              className="text-center px-4 py-3 font-mono font-semibold"
                              style={{ color: scoreColor(tech.overall ?? 0) }}
                            >
                              {tech.overall?.toFixed(3) ?? '0.000'}
                            </td>
                            <td className="text-center px-4 py-3">
                              <button
                                onClick={() =>
                                  setExpandedTechnique(
                                    expandedTechnique === tech.technique ? null : tech.technique
                                  )
                                }
                                className="text-xs font-medium hover:underline"
                                style={{ color: 'var(--blue)' }}
                              >
                                View &rarr;
                              </button>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </section>
        </div>
      )}
      {/* ═══ Technique Details Modal ═══ */}
      {expandedTechnique && expandedResult && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center pt-[8vh] overflow-y-auto"
          style={{ background: 'rgba(0,0,0,0.25)' }}
          onClick={() => setExpandedTechnique(null)}
        >
          <div
            className="relative w-full max-w-2xl rounded-lg shadow-xl mb-16"
            style={{ background: 'var(--surface)' }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div
              className="flex items-center justify-between px-6 py-4"
              style={{ borderBottom: '1px solid var(--border)' }}
            >
              <div className="flex items-center gap-3">
                <h2 className="text-lg font-semibold">Technique Details</h2>
                <span
                  className="font-mono text-xs px-2 py-1 rounded"
                  style={{ color: 'var(--text)', border: '1px solid var(--border)' }}
                >
                  {expandedTechnique.toUpperCase()}
                </span>
              </div>
              <button
                onClick={() => setExpandedTechnique(null)}
                className="w-8 h-8 flex items-center justify-center rounded-md transition-colors"
                style={{ color: 'var(--text-muted)' }}
              >
                &#x2715;
              </button>
            </div>

            <div className="p-6 space-y-6">
              {(expandedWasExtended || expandedStillTruncated) && (
                <div
                  className="px-3 py-2 rounded text-xs"
                  style={{
                    background: expandedStillTruncated ? '#fef2f2' : '#eff6ff',
                    border: `1px solid ${expandedStillTruncated ? '#fecaca' : '#bfdbfe'}`,
                    color: expandedStillTruncated ? '#dc2626' : 'var(--blue)',
                  }}
                >
                  {expandedStillTruncated
                    ? '⚠ This technique output may still be incomplete due token limits.'
                    : `↻ This technique used ${expandedContinuationRounds} continuation round${expandedContinuationRounds === 1 ? '' : 's'} to complete output.`}
                </div>
              )}

              {/* Performance */}
              <div>
                <p
                  className="text-[11px] font-mono uppercase tracking-wider mb-3"
                  style={{ color: 'var(--text-subtle)' }}
                >
                  Performance
                </p>
                <div className="grid grid-cols-4 gap-3">
                  {[
                    { label: 'OVERALL', value: expandedResult.scores?.overall },
                    { label: 'ACCURACY', value: expandedResult.scores?.accuracy },
                    { label: 'COMPLETENESS', value: expandedResult.scores?.completeness },
                    { label: 'EFFICIENCY', value: expandedResult.scores?.efficiency },
                  ].map((s) => (
                    <div
                      key={s.label}
                      className="p-3 rounded-md text-center"
                      style={{ border: '1px solid var(--border)' }}
                    >
                      <p
                        className="text-[9px] font-mono uppercase tracking-widest mb-1"
                        style={{ color: 'var(--text-subtle)' }}
                      >
                        {s.label}
                      </p>
                      <p
                        className="text-2xl font-mono font-light"
                        style={{ color: scoreColor(s.value ?? 0) }}
                      >
                        {s.value?.toFixed(3) ?? '0.000'}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Prompt */}
              <div>
                <p
                  className="text-[11px] font-mono uppercase tracking-wider mb-2"
                  style={{ color: 'var(--text-subtle)' }}
                >
                  Prompt Used
                </p>
                <div
                  className="p-4 rounded-md max-h-48 overflow-y-auto"
                  style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
                >
                  <pre className="text-sm font-mono whitespace-pre-wrap leading-relaxed">
                    {expandedResult.prompt}
                  </pre>
                </div>
              </div>

              {/* Response */}
              <div>
                <p
                  className="text-[11px] font-mono uppercase tracking-wider mb-2"
                  style={{ color: 'var(--text-subtle)' }}
                >
                  Model Response
                </p>
                <div
                  className="p-4 rounded-md max-h-72 overflow-y-auto"
                  style={{ background: '#f0fdf4', border: '1px solid #bbf7d0' }}
                >
                  <pre className="text-sm font-mono whitespace-pre-wrap leading-relaxed">
                    {expandedResult.response}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
