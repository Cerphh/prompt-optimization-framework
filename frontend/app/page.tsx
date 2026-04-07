'use client'

import { useState, useEffect } from 'react'

type RunMode = 'normal' | 'benchmark' | 'baseline'
type ScoreDisplayFormat = 'percent' | 'decimal'

interface BaselineResult {
  problem: string
  ground_truth_used: boolean
  runs_requested: number
  runs_succeeded: number
  run_mode: 'baseline'
  model_name: string
  technique: string
  prompt_used: string
  best_response: string
  scores: {
    accuracy: number
    consistency: number | null
    efficiency: number
    overall: number
    consistency_is_provisional: boolean
    consistency_runs_used: number
  }
  metrics: {
    elapsed_time: number
    total_tokens: number
    prompt_tokens: number
    completion_tokens: number
  }
  run_history: Array<{
    run_index: number
    success: boolean
    response?: string
    error?: string
    metrics: {
      elapsed_time: number
      total_tokens: number
      prompt_tokens: number
      completion_tokens: number
    }
    scores: {
      accuracy: number
      consistency: number | null
      efficiency: number
      overall: number
    }
  }>
}

interface Technique {
  technique: string
  accuracy: number
  consistency: number | null
  efficiency: number
  overall: number
  consistency_is_provisional?: boolean
  consistency_runs_used?: number
}

interface IndividualRun {
  technique: string
  run_index: number
  runs_configured: number
  success: boolean
  prompt?: string
  response?: string
  error?: string
  metrics: {
    elapsed_time: number
    total_tokens: number
    prompt_tokens: number
    completion_tokens: number
    done_reason?: string
    truncated?: boolean
    continuation_rounds?: number
    continuation_error?: string | null
    verifier_retry_applied?: boolean
    verifier_verdict?: string
  }
  normalized_output?: string
  scores: {
    accuracy: number
    consistency: number | null
    efficiency: number
    overall: number
    consistency_is_provisional: boolean
    consistency_runs_used: number
    consistency_matching_runs?: number | null
    overall_is_provisional: boolean
    overall_note?: string
  }
}

interface TechniqueResult {
  technique: string
  success: boolean
  prompt?: string
  response?: string
  error?: string
  metrics?: {
    elapsed_time: number
    total_tokens: number
    prompt_tokens: number
    completion_tokens: number
    truncated?: boolean
    continuation_rounds?: number
    runs_recorded?: number
    runs_succeeded?: number
    runs_failed?: number
  }
  scores: {
    accuracy: number
    consistency: number | null
    efficiency: number
    overall: number
    consistency_is_provisional?: boolean
    consistency_runs_used?: number
    consistency_matching_runs?: number | null
    overall_is_provisional?: boolean
    overall_note?: string
  }
  run_history?: IndividualRun[]
}

interface BenchmarkResult {
  problem: string
  runs_per_technique?: number
  best_technique: string
  best_result: TechniqueResult
  all_results: Record<string, TechniqueResult>
  comparison: Technique[]
  all_responses: Record<string, { response: string; score: number }>
  run_mode?: RunMode
  ground_truth_used?: boolean
  model_name?: string
  selection_source?:
    | 'db_history'
    | 'db_profile_rules'
    | 'runtime_scores'
  selection_details?: Record<string, unknown>
  pre_execution_policy?: {
    reason?: string
    history_source?: 'db_history' | 'db_profile_rules' | null
    selected_techniques?: string[]
    best_technique?: string | null
  }
  execution_summary?: {
    attempted_techniques?: string[]
    attempted_count?: number
    successful_count?: number
  }
  storage?: {
    success: boolean
    document_id?: string
    error?: string
  }
  few_shot_unavailable?: boolean
  few_shot_error?: string
}

interface TechniqueRow {
  technique: string
  success: boolean
  accuracy: number
  consistency: number
  consistencyAvailable: boolean
  consistencyRunsUsed: number
  consistencyIsProvisional: boolean
  efficiency: number
  overall: number
  overallIsProvisional: boolean
  metrics: {
    elapsed_time: number
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
  wordCount: number
  error?: string
}

const countWords = (value?: string): number => {
  const normalized = (value || '').trim()
  if (!normalized) {
    return 0
  }
  return normalized.split(/\s+/).length
}

const formatElapsedTime = (value: number): string => {
  if (!Number.isFinite(value) || value <= 0) {
    return '0.00s'
  }
  if (value >= 10) {
    return `${value.toFixed(1)}s`
  }
  return `${value.toFixed(2)}s`
}

const formatWholeNumber = (value: number): string => {
  if (!Number.isFinite(value)) {
    return '0'
  }
  return Math.round(value).toLocaleString()
}

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')
const INITIAL_RUNS_PER_TECHNIQUE = 1
const CONSISTENCY_TEST_RUNS_PER_TECHNIQUE = 3

const apiUrl = (path: string): string => `${API_BASE_URL}${path}`

/* Score color mapping */
const scoreColor = (v: number): string => {
  if (v >= 0.95) return 'var(--green)'
  if (v >= 0.80) return 'var(--blue)'
  if (v >= 0.65) return 'var(--amber)'
  return 'var(--text)'
}

const toSafeNumber = (value: unknown): number => {
  const parsed = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

const formatScorePercent = (
  value: unknown,
  fallback = 'N/A',
  digits = 1
): string => {
  if (value === null || value === undefined) {
    return fallback
  }
  const parsed = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(parsed)) {
    return fallback
  }
  const percentage = parsed * 100
  const rounded = Number(percentage.toFixed(digits))
  return `${rounded}%`
}

const formatScoreDecimal = (
  value: unknown,
  fallback = 'N/A',
  digits = 4
): string => {
  if (value === null || value === undefined) {
    return fallback
  }
  const parsed = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(parsed)) {
    return fallback
  }
  return parsed.toFixed(digits)
}

const formatScore = (
  value: unknown,
  format: ScoreDisplayFormat,
  fallback = 'N/A',
): string => {
  return format === 'percent'
    ? formatScorePercent(value, fallback)
    : formatScoreDecimal(value, fallback)
}

const HIDDEN_PROMPT_PREAMBLES = [
  "Solve the following math problems and give the final answer. Use the following examples only as style references. Do NOT repeat or copy any example answer. You must solve ONLY the target problem shown after 'TARGET PROBLEM'. Think carefully and use the examples only for internal reasoning. Output ONLY the final answer for the TARGET PROBLEM. Do NOT include steps, explanations, or extra text.",
  "Solve the following math problem and end with a concise final answer. Do NOT show steps or explanations.",
]

const getDisplayPrompt = (prompt?: string): string => {
  if (!prompt) return 'No prompt available'

  let display = prompt
  for (const preamble of HIDDEN_PROMPT_PREAMBLES) {
    if (display.startsWith(preamble)) {
      display = display.slice(preamble.length).trimStart()
    }
  }

  return display.trim() || 'No prompt available'
}

type SelectionSource = 'db_profile_rules' | 'db_history' | 'runtime_scores'

interface TierInfo {
  tier: number
  tierName: string
  strategyName: string
  description: string
  bgColor: string
  borderColor: string
  textColor: string
}

const getTierInfo = (source?: SelectionSource): TierInfo => {
  const tierMap: Record<SelectionSource, TierInfo> = {
    db_profile_rules: {
      tier: 1,
      tierName: 'Profile Based Selection',
      strategyName: 'Profile-Based Selection',
      description: 'Selected based on similar problem profiles using weighted similarity scoring. This is the most intelligent selection strategy.',
      bgColor: '#f0fdf4',
      borderColor: '#bbf7d0',
      textColor: 'var(--green)',
    },
    db_history: {
      tier: 2,
      tierName: 'Domain-Average Fallback',
      strategyName: 'Domain-Average Fallback',
      description: 'Selected based on historical domain/difficulty averages. Used when profile-based selection lacked sufficient confidence.',
      bgColor: '#eff6ff',
      borderColor: '#bfdbfe',
      textColor: 'var(--blue)',
    },
    runtime_scores: {
      tier: 3,
      tierName: 'Runtime Selection',
      strategyName: 'Runtime Selection',
      description: 'Selected based on live execution scores. All techniques executed and compared at runtime.',
      bgColor: '#f5f5f4',
      borderColor: '#d6d3d1',
      textColor: '#57534e',
    },
  }

  return source && source in tierMap 
    ? tierMap[source]
    : {
        tier: 0,
        tierName: 'Unknown',
        strategyName: 'Selection Strategy',
        description: 'Selection strategy information not available.',
        bgColor: 'var(--surface)',
        borderColor: 'var(--border)',
        textColor: 'var(--text-muted)',
      }
}

const formatSelectionReason = (reason?: string): string => {
  if (!reason) return 'No reason provided'
  
  const reasonMap: Record<string, string> = {
    'db_confidence_rules': 'Did not meet confidence thresholds required by DB selection rules',
    'exploration_runtime': 'Forced runtime selection for exploration to gather diverse data',
    'profile_best_missing_result': 'Profile-recommended technique failed during execution',
    'low_confidence_gap': 'Recommendation confidence was too low, fell back to domain average',
    'low_technique_confidence': 'Top technique has inconsistent scores (low confidence), fell back',
    'insufficient_samples': 'Insufficient historical data for confident recommendation',
    'no_profile_match': 'No similar problems found in database history',
    'missing_problem_profile': 'Could not extract problem profile',
  }

  return reasonMap[reason] || reason
}

interface TierDecisionInfo {
  tierNumber: number
  tierName: string
  wasAttempted: boolean
  result: 'passed' | 'failed' | 'skipped'
  reason?: string
  details?: string
}

const buildDecisionTree = (details?: Record<string, any>, source?: SelectionSource): TierDecisionInfo[] => {
  if (!details) return []
  
  const profileSelection = details.profile_selection as Record<string, any> || {}
  const domainSelection = details.domain_selection as Record<string, any> || {}
  const profileReason = details.profile_decision_reason || profileSelection.reason || ''
  const dbReason = details.db_decision_reason || domainSelection.reason || ''
  const profileAttempted = profileSelection.success !== undefined
  const domainAttempted = domainSelection.success !== undefined
  const profileSelected = source === 'db_profile_rules'
  const domainSelected = source === 'db_history'
  const runtimeSelected = source === 'runtime_scores'

  const tree: TierDecisionInfo[] = [
    {
      tierNumber: 1,
      tierName: 'Profile Based Selection',
      wasAttempted: profileAttempted,
      result: profileSelected ? 'passed' : profileAttempted ? 'failed' : 'skipped',
      reason: profileReason,
      details: buildTier1Details(profileSelection, details),
    },
    {
      tierNumber: 2,
      tierName: 'Tier 2: Domain-Average Fallback',
      wasAttempted: domainAttempted,
      result: domainSelected ? 'passed' : domainAttempted ? 'failed' : 'skipped',
      reason: dbReason,
      details: buildTier2Details(domainSelection, details),
    },
    {
      tierNumber: 3,
      tierName: 'Runtime Selection',
      wasAttempted: true,
      result: runtimeSelected ? 'passed' : 'skipped',
      reason: details.reason,
      details: 'All techniques executed; best selected by runtime performance scores.',
    },
  ]

  return tree
}

const buildTier1Details = (prof: Record<string, any>, allDetails: Record<string, any>): string => {
  if (!prof.success) {
    const reason = prof.reason || 'unknown'
    if (reason === 'no_profile_match') {
      return `No similar problems in database (similarity < ${allDetails.db_confidence_rules?.profile_min_similarity ? (allDetails.db_confidence_rules.profile_min_similarity * 100).toFixed(0) : '50'}%)`
    }
    if (reason === 'missing_problem_profile') {
      return 'Could not extract problem features (intent, math features, format, constraints)'
    }
    return `Profile lookup failed: ${reason}`
  }

  const rules = allDetails.db_confidence_rules || {}
  const ranking = prof.ranking || []
  
  if (ranking.length === 0) return 'No ranking data available'

  const top = ranking[0] as Record<string, any>
  const topSamples = top.samples || top.unweighted_samples || top.effective_samples || 0
  const minSamples = rules.profile_min_samples_per_technique || 3

  if (topSamples < minSamples) {
    return `❌ Top technique has ${topSamples} sample(s) — needs ≥${minSamples}`
  }

  if (ranking.length > 1) {
    const second = ranking[1] as Record<string, any>
    const topScore = top.weighted_average || top.average_overall || 0
    const secondScore = second.weighted_average || second.average_overall || 0
    const gap = topScore - secondScore
    const baseGap = rules.profile_min_average_gap || 0.005
    const dataSamples = Math.max(topSamples, second.samples || second.unweighted_samples || 0)

    // Adaptive thresholds: more data → stricter gap detection & wider tie zone
    const effectiveGap = dataSamples >= 10 ? baseGap * 0.5 : dataSamples >= 5 ? baseGap * 0.75 : baseGap
    const indistinguishableThreshold = dataSamples >= 10 ? 0.008 : dataSamples >= 5 ? 0.004 : 0.003

    if (gap < effectiveGap) {
      if (gap < indistinguishableThreshold) {
        return `✓ Both techniques scored equally — ${ranking[0].technique} selected (${dataSamples} samples, indistinguishable)`
      }
      return `❌ Top vs 2nd gap ${(gap * 100).toFixed(2)}% — needs ≥${(effectiveGap * 100).toFixed(2)}% (adaptive, ${dataSamples} samples)`
    }
  }

  return `✓ Passed all checks — ${ranking[0].technique} matched ${prof.matched_documents} similar problems`
}

const buildTier2Details = (dom: Record<string, any>, allDetails: Record<string, any>): string => {
  if (!dom.success) {
    const reason = dom.reason || 'unknown'
    return `❌ No domain data available: ${reason}`
  }

  const rules = allDetails.db_confidence_rules || {}
  const ranking = dom.ranking || []
  const top = ranking[0] as Record<string, any> || {}
  const samples = top.samples || 0
  const minSamples = rules.min_samples_per_technique || 3

  if (samples < minSamples) {
    return `❌ Domain has ${samples} sample(s) — needs ≥${minSamples}`
  }

  if (ranking.length > 1) {
    const second = ranking[1] as Record<string, any>
    const topScore = top.average_overall || 0
    const secondScore = second.average_overall || 0
    const gap = topScore - secondScore
    const baseGap = rules.min_average_gap || 0.05
    const dataSamples = Math.max(samples, second.samples || 0)

    const effectiveGap = dataSamples >= 10 ? baseGap * 0.5 : dataSamples >= 5 ? baseGap * 0.75 : baseGap
    const indistinguishableThreshold = dataSamples >= 10 ? 0.008 : dataSamples >= 5 ? 0.004 : 0.003

    if (gap < effectiveGap) {
      if (gap < indistinguishableThreshold) {
        return `✓ Both techniques scored equally — ${ranking[0]?.technique || 'technique'} selected (${dataSamples} samples, indistinguishable)`
      }
      return `❌ Top vs 2nd gap ${(gap * 100).toFixed(2)}% — needs ≥${(effectiveGap * 100).toFixed(2)}% (adaptive, ${dataSamples} samples)`
    }
  }

  return `✓ Passed all checks — ${ranking[0]?.technique || 'technique'} based on domain/${allDetails.difficulty} history`
}

interface HistoricalStats {
  winRate: number | null
  avgScore: number | null
  samples: number
  lead: number | null
  ranking: { technique: string; avgScore: number; winRate: number }[]
}

const getHistoricalStats = (
  source?: SelectionSource,
  details?: Record<string, any>,
): HistoricalStats => {
  const empty: HistoricalStats = { winRate: null, avgScore: null, samples: 0, lead: null, ranking: [] }
  if (!details) return empty

  let ranking: Record<string, any>[] = []

  if (source === 'db_profile_rules') {
    const prof = details.profile_selection as Record<string, any> || {}
    ranking = (prof.ranking || []) as Record<string, any>[]
  } else if (source === 'db_history') {
    const dom = details.domain_selection as Record<string, any> || {}
    ranking = (dom.ranking || []) as Record<string, any>[]
  }

  if (ranking.length === 0) return empty

  const top = ranking[0]
  const winRate = top.win_rate ?? null
  const avgScore = top.weighted_average ?? top.average_overall ?? null
  const samples = top.samples || top.unweighted_samples || top.effective_samples || 0

  let lead: number | null = null
  if (ranking.length > 1) {
    const second = ranking[1]
    const topScore = top.weighted_average ?? top.average_overall ?? 0
    const secondScore = second.weighted_average ?? second.average_overall ?? 0
    lead = topScore - secondScore
  }

  const parsedRanking = ranking.map((r: Record<string, any>) => ({
    technique: (r.technique || '').toUpperCase(),
    avgScore: r.weighted_average ?? r.average_overall ?? 0,
    winRate: r.win_rate ?? 0,
  }))

  return { winRate, avgScore, samples, lead, ranking: parsedRanking }
}

const buildTierSummary = (
  source: SelectionSource,
  details: Record<string, any>,
  bestTechnique?: string,
): string => {
  const profileSelection = details.profile_selection as Record<string, any> || {}
  const domainSelection = details.domain_selection as Record<string, any> || {}
  const technique = bestTechnique?.toUpperCase() || 'N/A'

  if (source === 'db_profile_rules') {
    const matched = profileSelection.matched_documents
    const ranking = profileSelection.ranking || []
    const top = ranking[0] as Record<string, any> || {}
    const samples = top.samples || top.unweighted_samples || matched || 0
    return `${technique} selected from ${samples} similar problem${samples !== 1 ? 's' : ''} in history`
  }

  if (source === 'db_history') {
    const ranking = domainSelection.ranking || []
    const top = ranking[0] as Record<string, any> || {}
    const samples = top.samples || 0
    const domain = domainSelection.domain || 'domain'
    const difficulty = domainSelection.difficulty || 'basic'
    return `${technique} selected from ${samples} sample${samples !== 1 ? 's' : ''} in ${domain}/${difficulty}`
  }

  return 'All techniques executed, best selected by runtime scores'
}


export default function Home() {
  const [problem, setProblem] = useState('')
  const [subject, setSubject] = useState('algebra')
  const [difficulty, setDifficulty] = useState('basic')
  const [runMode, setRunMode] = useState<RunMode>('normal')
  const [difficultyManualOverride, setDifficultyManualOverride] = useState(false)
  const [groundTruth, setGroundTruth] = useState('')
  const [lastRunUsedGroundTruth, setLastRunUsedGroundTruth] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<BenchmarkResult | null>(null)
  const [baselineResult, setBaselineResult] = useState<BaselineResult | null>(null)
  const [error, setError] = useState('')
  const [isScopeError, setIsScopeError] = useState(false)
  const [expandedTechnique, setExpandedTechnique] = useState<string | null>(null)
  const [efficiencyInfo, setEfficiencyInfo] = useState<{ technique: string; x: number; y: number } | null>(null)
  const [showIndividualRuns, setShowIndividualRuns] = useState(false)
  const [healthStatus, setHealthStatus] = useState<'checking' | 'healthy' | 'unhealthy'>('checking')
  const [validationError, setValidationError] = useState('')
  const [savingToDb, setSavingToDb] = useState(false)
  const [saveStatus, setSaveStatus] = useState('')
  const [didSaveToDb, setDidSaveToDb] = useState(false)
  const [didExportJson, setDidExportJson] = useState(false)
  const [savingBaselineToDb, setSavingBaselineToDb] = useState(false)
  const [baselineSaveStatus, setBaselineSaveStatus] = useState('')
  const [didSaveBaselineToDb, setDidSaveBaselineToDb] = useState(false)
  const [streamingResponse, setStreamingResponse] = useState('')
  const [streamingTechnique, setStreamingTechnique] = useState('')
  const [streamingStatus, setStreamingStatus] = useState('')
  const [perfScoreFormat, setPerfScoreFormat] = useState<ScoreDisplayFormat>('percent')
  const [compScoreFormat, setCompScoreFormat] = useState<ScoreDisplayFormat>('percent')
  const [benchmarkRuns, setBenchmarkRuns] = useState(CONSISTENCY_TEST_RUNS_PER_TECHNIQUE)
  const [coverageData, setCoverageData] = useState<{ problem_count: number; ground_truth_count: number; techniques_tested: string[]; win_counts: Record<string, number>; total_with_winner: number } | null>(null)
  const [exampleTypes, setExampleTypes] = useState<Record<string, Record<string, { concept: string; count: number; sample: string }[]>>>({})
  const [showExampleTypes, setShowExampleTypes] = useState(false)
  const [activeConceptTab, setActiveConceptTab] = useState<string>('')
  const [availableDifficulties, setAvailableDifficulties] = useState<Record<string, string[]>>({})

  // ── Add Example state ──
  const [showAddExample, setShowAddExample] = useState(false)
  const [addExProblem, setAddExProblem] = useState('')
  const [addExSolution, setAddExSolution] = useState('')
  const [addExSubject, setAddExSubject] = useState('')
  const [addExDifficulty, setAddExDifficulty] = useState('')
  const [addExType, setAddExType] = useState('')
  const [addExConcept, setAddExConcept] = useState('')
  const [addExAnalyzed, setAddExAnalyzed] = useState(false)
  const [addExAnalyzing, setAddExAnalyzing] = useState(false)
  const [addExSaving, setAddExSaving] = useState(false)
  const [addExStatus, setAddExStatus] = useState('')
  const [addExExistingTypes, setAddExExistingTypes] = useState<Record<string, string[]>>({})
  const [addExDetectionMethod, setAddExDetectionMethod] = useState('')
  // Example 2
  const [addExProblem2, setAddExProblem2] = useState('')
  const [addExSolution2, setAddExSolution2] = useState('')
  const [addExSubject2, setAddExSubject2] = useState('')
  const [addExDifficulty2, setAddExDifficulty2] = useState('')
  const [addExType2, setAddExType2] = useState('')
  const [addExConcept2, setAddExConcept2] = useState('')
  const [addExAnalyzed2, setAddExAnalyzed2] = useState(false)
  const [addExDetectionMethod2, setAddExDetectionMethod2] = useState('')
  // Fetch available difficulties per subject from example_problems.json
  useEffect(() => {
    const fetchDifficulties = async () => {
      try {
        const res = await fetch(apiUrl('/example-difficulties'))
        if (res.ok) {
          const data = await res.json()
          setAvailableDifficulties(data)
        }
      } catch { /* ignore */ }
    }
    fetchDifficulties()
  }, [])

  // When subject changes, default to basic if the current difficulty isn't in the JSON (but still allow override)
  useEffect(() => {
    const diffs = availableDifficulties[subject]
    if (!difficultyManualOverride && diffs && diffs.length > 0 && !diffs.includes(difficulty)) {
      setDifficulty('basic')
    }
  }, [subject, availableDifficulties])

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

  // Fetch available few-shot example types
  useEffect(() => {
    const fetchExampleTypes = async () => {
      try {
        const res = await fetch(apiUrl('/example-types'))
        if (res.ok) {
          const data = await res.json()
          setExampleTypes(data)
        }
      } catch {
        // silent – non-critical
      }
    }
    fetchExampleTypes()
  }, [])

  useEffect(() => {
    if (runMode !== 'benchmark' && validationError === 'Benchmark mode requires an expected answer') {
      setValidationError('')
    }
  }, [runMode, validationError])

  const formatTypeName = (t: string) => t.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

  const renderErrorPanel = () => {
    if (!error) return null

    if (isScopeError) {
      return (
        <div
          className="mt-5 p-4 rounded-md text-sm leading-relaxed"
          style={{ background: '#fff7ed', border: '1px solid #fed7aa', color: '#9a3412' }}
        >
          <p className="font-semibold mb-1">Outside Framework Scope</p>
          <p>{error}</p>
          <p className="mt-2 text-xs" style={{ color: '#c2410c' }}>
            Supported domains: Algebra, Pre-Calculus, Counting &amp; Probability.
          </p>
        </div>
      )
    }

    return (
      <div
        className="mt-5 p-3 rounded-md text-xs leading-relaxed whitespace-pre-wrap"
        style={{ background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b' }}
      >
        {error}
      </div>
    )
  }

  // ── Add Example helpers ──
  const analyzeExample = async () => {
    if (!addExProblem.trim()) return
    setAddExAnalyzing(true)
    setAddExStatus('')
    try {
      // Analyze example 1
      const res = await fetch(apiUrl('/examples/analyze'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ problem: addExProblem, solution: addExSolution }),
      })
      if (!res.ok) throw new Error('Analysis failed')
      const data = await res.json()
      setAddExSubject(data.detected_subject)
      setAddExDifficulty(data.detected_difficulty)
      setAddExType(data.detected_type)
      setAddExConcept(data.detected_concept || data.detected_type)
      setAddExExistingTypes(data.existing_types || {})
      setAddExDetectionMethod(data.detection_method || 'rule-based')
      setAddExAnalyzed(true)

      // Analyze example 2 if filled
      if (addExProblem2.trim() && addExSolution2.trim()) {
        const res2 = await fetch(apiUrl('/examples/analyze'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ problem: addExProblem2, solution: addExSolution2 }),
        })
        if (res2.ok) {
          const data2 = await res2.json()
          setAddExSubject2(data2.detected_subject)
          setAddExDifficulty2(data2.detected_difficulty)
          setAddExType2(data2.detected_type)
          setAddExConcept2(data2.detected_concept || data2.detected_type)
          setAddExDetectionMethod2(data2.detection_method || 'rule-based')
          setAddExAnalyzed2(true)
        }
      }
    } catch (e: unknown) {
      setAddExStatus(`Analysis error: ${e instanceof Error ? e.message : 'Unknown error'}`)
    } finally {
      setAddExAnalyzing(false)
    }
  }

  const saveExample = async () => {
    if (!addExProblem.trim() || !addExSolution.trim()) {
      setAddExStatus('Problem and solution are required')
      return
    }
    setAddExSaving(true)
    setAddExStatus('')
    try {
      // Save example 1
      const res = await fetch(apiUrl('/examples'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          problem: addExProblem,
          solution: addExSolution,
          subject: addExSubject,
          difficulty: addExDifficulty,
          type: addExType,
          concept: addExConcept,
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Save failed')

      let savedCount = 1

      // Save example 2 if filled and analyzed
      if (addExProblem2.trim() && addExSolution2.trim() && addExAnalyzed2) {
        const res2 = await fetch(apiUrl('/examples'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            problem: addExProblem2,
            solution: addExSolution2,
            subject: addExSubject2,
            difficulty: addExDifficulty2,
            type: addExType2,
            concept: addExConcept2,
          }),
        })
        const data2 = await res2.json()
        if (!res2.ok) throw new Error(data2.detail || 'Save failed for example 2')
        savedCount = 2
      }

      setAddExStatus(`Saved ${savedCount} example${savedCount > 1 ? 's' : ''}!`)
      // Reset form after success
      setTimeout(() => {
        setAddExProblem(''); setAddExSolution('')
        setAddExProblem2(''); setAddExSolution2('')
        setAddExAnalyzed(false); setAddExAnalyzed2(false)
        setAddExStatus('')
        setShowAddExample(false)
        // Refresh example types
        fetch(apiUrl('/example-types')).then(r => r.json()).then(d => setExampleTypes(d)).catch(() => {})
      }, 1500)
    } catch (e: unknown) {
      setAddExStatus(`Error: ${e instanceof Error ? e.message : 'Unknown error'}`)
    } finally {
      setAddExSaving(false)
    }
  }

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

    // Exclude pre-calculus / trig / vector / matrix problems
    const explicitPrecalculus = /\bd\/dx\b|\bdy\/dx\b|∫|\bderivative\b|\bdifferentiate\b|\bintegral\b|\bintegrate\b|\blimit\b|\blim\b|\bsin\b|\bcos\b|\btan\b|\bsec\b|\bcsc\b|\bcot\b|\barcsin\b|\barccos\b|\barctan\b|\bslope\b|\bconvert\b|\bradian|\bdegree|\bmatrix\b|\bdet\b|\bsingular\b|\bnon-invertible\b|\binvertible\b|\bdeterminant\b|\borthogonal\b|\bperpendicular\b|\bvector|\bunit vector|\b[⟨<]|\btriangle\b|△|\blaw of sines\b|\blaw of cosines\b|\br\s*\(\s*t\s*\)/.test(lowerText)
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
                 'trigonometric', 'sine', 'cosine', 'tangent function', 'periodic',
                 'radian', 'degrees', 'convert', 'angle', 'unit circle',
                 'vector', 'orthogonal', 'perpendicular', 'dot product', 'magnitude',
                 'matrix', 'determinant', 'singular', 'non-invertible', 'invertible',
                 'law of sines', 'law of cosines', 'triangle', 'parametric',
                 'harmonic', 'identity', 'double angle'],
      patterns: [/d\/dx/, /∫/, /lim|limit/, /derivative|derivatives/, /integral|integrate/i, 
             /exponential|logarithm|log\(/, /sin|cos|tan|graph|function/, /find.*(derivative|integral|limit|domain|range|asymptote)/,
             /\bslope\b.*through/, /convert.*(?:degree|radian)/, /evaluate\s+(?:sin|cos|tan)/,
             /maximum\s+value\s+of\s+\d+\s*(?:cos|sin)/, /orthogonal|perpendicular/,
             /matrix.*(?:singular|invertible|determinant)/, /△|\\triangle/,
             /\br\s*\(\s*t\s*\)/, /unit\s+vector/, /⟨|⟩/],
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
      keywords: ['solve', 'solve for', 'factor', 'factorize', 'expand', 'quadratic', 
                 'equation', 'polynomial', 'linear', 'system of equations', 'roots', 'zero', 'parabola',
                 'binomial', 'trinomial', 'monomial', 'rational', 'radical', 'algebraic', 'real solutions', 'real roots',
                 'substitute', 'inequality', 'variable', 'coefficient'],
      patterns: [/solve (for|x|y|z)/, /factor|factorize/, /expand/, /quadratic/i,
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
        const diffs = availableDifficulties[detectedSubject]
        if (diffs && diffs.length > 0 && !diffs.includes(detectedDifficulty)) {
          setDifficulty('basic')  // default to basic if not in JSON
        } else {
          setDifficulty(detectedDifficulty)
        }
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

  const runBenchmarkFlow = async (
    runsPerTechnique: number,
    mode: 'initial' | 'consistency'
  ) => {
    // Validate before submitting
    if (problem.length < 10) {
      setValidationError('Problem must be at least 10 characters')
      return
    }

    if (runMode === 'benchmark' && !groundTruth.trim()) {
      setValidationError('Benchmark mode requires an expected answer')
      return
    }

    setLoading(true)
    setError('')
    setIsScopeError(false)
    setSaveStatus('')
    setDidSaveToDb(false)
    setDidExportJson(false)
    setBaselineSaveStatus('')
    setDidSaveBaselineToDb(false)
    setResult(null)
    setBaselineResult(null)
    setExpandedTechnique(null)
    setStreamingResponse('')
    setStreamingTechnique('')
    setCoverageData(null)
    setStreamingStatus(
      mode === 'consistency'
        ? `Running consistency test (${runsPerTechnique} runs per technique)...`
        : runMode === 'benchmark'
        ? 'Initializing benchmark mode (single-run scoring)...'
        : 'Initializing normal mode (single-run scoring)...'
    )

    try {
      // Check connection first
      if (healthStatus === 'unhealthy') {
        throw new Error('⚠️ System offline: Make sure Ollama is running (ollama serve) and the backend API is started')
      }

      const groundTruthValue = runMode === 'benchmark' ? groundTruth.trim() : ''
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
          run_mode: runMode,
          speed_profile: 'balanced',
          runs_per_technique: runsPerTechnique,
          ...(groundTruthValue ? { ground_truth: groundTruthValue } : {}),
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        const detail =
          typeof errorData.detail === 'string'
            ? errorData.detail
            : JSON.stringify(errorData.detail || 'Unknown error')

        if (
          response.status === 422
          && detail.toLowerCase().includes('only supports the following 3 domains')
        ) {
          throw new Error(`SCOPE_ERROR:${detail}`)
        }

        if (response.status === 500) {
          throw new Error(detail)
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
            throw new Error(event.error || 'Unknown streaming error')
          }
        }
      }

      if (streamedText) {
        setStreamingResponse(streamedText)
      }

      if (!finalResult) {
        throw new Error('Benchmark stream ended before returning final result')
      }

      // For normal mode, derive selection_source from the pre-execution
      // policy (the single Tier 1/2 decision point).  The backend no
      // longer runs a redundant post-execution Tier check.
      if (finalResult.pre_execution_policy?.history_source) {
        finalResult.selection_source = finalResult.pre_execution_policy.history_source
        // Merge pre-execution details into selection_details so tier
        // display helpers (buildDecisionTree, buildTierSummary, etc.)
        // can read profile_selection / domain_selection.
        finalResult.selection_details = {
          ...finalResult.selection_details,
          ...finalResult.pre_execution_policy,
        }
      }

      setResult(finalResult)
      if (mode === 'consistency') {
        setSaveStatus('Consistency test complete. Overall score now includes consistency.')
      } else {
        setSaveStatus('Not saved yet. Click Save to DB to save this result.')
      }

      // Fetch coverage for normal mode
      if (runMode === 'normal') {
        fetchCoverage(subject, difficulty)
      }
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setIsScopeError(false)
        setError('🔴 Cannot connect to API. Make sure backend is running on port 8000')
      } else if (err instanceof Error && err.message.startsWith('SCOPE_ERROR:')) {
        setIsScopeError(true)
        setError(err.message.replace('SCOPE_ERROR:', '').trim())
      } else {
        setIsScopeError(false)
        if (err instanceof Error && err.message.includes('No prompting techniques available')) {
          setError('No matching few-shot examples found for this problem')
        } else {
          setError(err instanceof Error ? err.message : 'An error occurred')
        }
      }
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (runMode === 'baseline') {
      await runBaselineFlow()
    } else if (runMode === 'benchmark') {
      await runBenchmarkFlow(benchmarkRuns, benchmarkRuns >= CONSISTENCY_TEST_RUNS_PER_TECHNIQUE ? 'consistency' : 'initial')
    } else {
      await runBenchmarkFlow(INITIAL_RUNS_PER_TECHNIQUE, 'initial')
    }
  }

  const runBaselineFlow = async () => {
    if (problem.length < 10) {
      setValidationError('Problem must be at least 10 characters')
      return
    }

    setLoading(true)
    setError('')
    setIsScopeError(false)
    setSaveStatus('')
    setBaselineSaveStatus('')
    setDidSaveBaselineToDb(false)
    setResult(null)
    setBaselineResult(null)
    setExpandedTechnique(null)
    setStreamingResponse('')
    setStreamingTechnique('')
    setStreamingStatus('Running raw baseline (no prompting technique)...')

    try {
      if (healthStatus === 'unhealthy') {
        throw new Error('⚠️ System offline: Make sure Ollama is running (ollama serve) and the backend API is started')
      }

      const groundTruthValue = groundTruth.trim() || undefined

      const response = await fetch(apiUrl('/baseline'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          problem,
          runs: benchmarkRuns,
          subject,
          difficulty,
          ...(groundTruthValue ? { ground_truth: groundTruthValue } : {}),
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        const detail =
          typeof errorData.detail === 'string'
            ? errorData.detail
            : JSON.stringify(errorData.detail || 'Unknown error')
        throw new Error(`🔴 HTTP ${response.status}: ${detail}`)
      }

      const data: BaselineResult = await response.json()
      setBaselineResult(data)
      setLastRunUsedGroundTruth(data.ground_truth_used)
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

  const handleRunConsistencyTest = async () => {
    await runBenchmarkFlow(CONSISTENCY_TEST_RUNS_PER_TECHNIQUE, 'consistency')
  }

  const handleSaveBaselineToDb = async () => {
    if (!baselineResult) return

    setSavingBaselineToDb(true)
    setBaselineSaveStatus('')

    try {
      const response = await fetch(apiUrl('/baseline/save'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          result: baselineResult,
          source: 'frontend_manual_save',
          metadata: {
            subject,
            difficulty,
            domain: subject,
            has_ground_truth: baselineResult.ground_truth_used,
          },
        }),
      })

      const data = await response.json().catch(() => ({}))

      if (!response.ok || !data?.storage) {
        throw new Error(data?.detail || data?.storage?.error || 'Failed to save baseline to DB')
      }

      if (data.storage.success) {
        setDidSaveBaselineToDb(true)
        setBaselineSaveStatus('Saved to DB successfully.')
      } else {
        setBaselineSaveStatus(`Save failed: ${data.storage.error || 'Unknown error'}`)
      }
    } catch (err) {
      setBaselineSaveStatus(err instanceof Error ? err.message : 'Failed to save baseline to DB')
    } finally {
      setSavingBaselineToDb(false)
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
            run_mode: result.run_mode || runMode,
            has_ground_truth: result.ground_truth_used ?? lastRunUsedGroundTruth,
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

  const fetchCoverage = async (dom: string, diff: string) => {
    try {
      const response = await fetch(apiUrl(`/coverage/${encodeURIComponent(dom)}/${encodeURIComponent(diff)}`))
      if (response.ok) {
        const data = await response.json()
        if (data.success) setCoverageData(data)
      }
    } catch { /* ignore */ }
  }
  const bestContinuationRounds = bestMetrics?.continuation_rounds ?? 0
  const bestWasExtended = bestContinuationRounds > 0
  const bestStillTruncated = bestMetrics?.truncated === true

  const expandedMetrics = expandedResult?.metrics
  const expandedContinuationRounds = expandedMetrics?.continuation_rounds ?? 0
  const expandedWasExtended = expandedContinuationRounds > 0
  const expandedStillTruncated = expandedMetrics?.truncated === true

  const shouldShowOnlySelectedTechnique =
    result?.selection_source === 'db_profile_rules' || result?.selection_source === 'db_history'

  const visibleTechniqueEntries = result
    ? Object.entries(result.all_results || {}).filter(([techniqueName]) => {
        if (!shouldShowOnlySelectedTechnique) {
          return true
        }
        return techniqueName === result.best_technique
      })
    : []

  const techniqueRows: TechniqueRow[] = result
    ? visibleTechniqueEntries.map(([techniqueName, techniqueResult]) => {
        const scores = techniqueResult?.scores || ({} as Partial<TechniqueResult['scores']>)
        const metrics = techniqueResult?.metrics
        return {
          technique: techniqueName,
          success: techniqueResult?.success ?? false,
          accuracy: toSafeNumber(scores.accuracy),
          consistency: toSafeNumber(scores.consistency),
          consistencyAvailable: scores.consistency !== null && scores.consistency !== undefined,
          consistencyRunsUsed: toSafeNumber(scores.consistency_runs_used),
          consistencyIsProvisional: Boolean(scores.consistency_is_provisional),
          efficiency: toSafeNumber(scores.efficiency),
          overall: toSafeNumber(scores.overall),
          overallIsProvisional: Boolean(scores.overall_is_provisional),
          metrics: {
            elapsed_time: toSafeNumber(metrics?.elapsed_time),
            prompt_tokens: toSafeNumber(metrics?.prompt_tokens),
            completion_tokens: toSafeNumber(metrics?.completion_tokens),
            total_tokens: toSafeNumber(metrics?.total_tokens),
          },
          wordCount: countWords(techniqueResult?.response),
          error: techniqueResult?.error,
        }
      })
    : []

  techniqueRows.sort((a, b) => {
    if (a.success !== b.success) {
      return Number(b.success) - Number(a.success)
    }
    if (a.overall !== b.overall) {
      return b.overall - a.overall
    }
    return a.technique.localeCompare(b.technique)
  })

  const activeMode: RunMode = baselineResult ? 'baseline' : (result?.run_mode || runMode) as RunMode
  const runUsedGroundTruth = baselineResult?.ground_truth_used ?? result?.ground_truth_used ?? lastRunUsedGroundTruth
  const modeLabel = activeMode === 'benchmark' ? 'Benchmark mode' : activeMode === 'baseline' ? 'Baseline mode' : 'Normal mode'
  const modeHint =
    activeMode === 'benchmark'
      ? 'Live comparison of all techniques with required expected answer.'
      : activeMode === 'baseline'
      ? 'Raw LLM — no instruction framing or prompt engineering applied.'
      : 'Uses historical preselection after enough data; otherwise runs all techniques.'
  const attemptedTechniqueCount = techniqueRows.length
  const successfulTechniqueCount = techniqueRows.filter((row) => row.success).length
  const canManageFewShotExamples = runMode !== 'baseline'

  const isDisabled =
    loading ||
    !!validationError ||
    problem.length < 10 ||
    (runMode === 'benchmark' && !groundTruth.trim())

  const canRunConsistencyTest =
    !loading &&
    !validationError &&
    problem.length >= 10 &&
    (runMode !== 'benchmark' || !!groundTruth.trim())

  const bestConsistencyIsProvisional =
    result?.best_result?.scores?.consistency_is_provisional === true

  const renderEfficiencyCell = (tech: TechniqueRow) => {
    const isOpen = efficiencyInfo?.technique === tech.technique

    return (
      <div className="inline-flex items-center justify-center gap-1.5">
        <span className="font-mono">{formatScore(tech.efficiency, compScoreFormat)}</span>
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation()
            if (isOpen) {
              setEfficiencyInfo(null)
            } else {
              const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
              setEfficiencyInfo({ technique: tech.technique, x: rect.right + 8, y: rect.top - 4 })
            }
          }}
          className="inline-flex h-[18px] w-[18px] items-center justify-center rounded-full text-[9px] font-semibold leading-none transition-colors shrink-0"
          style={{
            background: isOpen ? 'var(--accent)' : 'transparent',
            color: isOpen ? '#fff' : 'var(--text-subtle)',
            border: isOpen ? '1px solid var(--accent)' : '1px solid var(--border)',
          }}
          aria-label={`Show efficiency breakdown for ${tech.technique}`}
          title="Efficiency breakdown"
        >
          i
        </button>
      </div>
    )
  }

  const renderEfficiencyPopover = () => {
    if (!efficiencyInfo) return null
    const tech = techniqueRows.find((r) => r.technique === efficiencyInfo.technique)
    if (!tech) return null

    // Clamp so the popover stays on screen (popover is ~240px wide, ~200px tall)
    const popoverWidth = 240
    const popoverHeight = 210
    const viewportW = typeof window !== 'undefined' ? window.innerWidth : 1200
    const viewportH = typeof window !== 'undefined' ? window.innerHeight : 800
    const left = Math.min(efficiencyInfo.x, viewportW - popoverWidth - 12)
    const top = Math.min(Math.max(efficiencyInfo.y, 8), viewportH - popoverHeight - 12)

    return (
      <div
        className="fixed inset-0 z-50"
        onClick={() => setEfficiencyInfo(null)}
      >
        <div
          className="fixed w-60 rounded-lg p-4 shadow-xl"
          style={{
            left: `${left}px`,
            top: `${top}px`,
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            color: 'var(--text)',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between mb-3">
            <p className="text-[10px] font-mono uppercase tracking-widest" style={{ color: 'var(--text-subtle)' }}>
              Efficiency Breakdown
            </p>
            <button
              type="button"
              onClick={() => setEfficiencyInfo(null)}
              className="text-xs leading-none px-1"
              style={{ color: 'var(--text-subtle)' }}
            >
              ✕
            </button>
          </div>
          <p className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
            {tech.technique.toUpperCase()} · {formatScore(tech.efficiency, compScoreFormat)}
          </p>
          <div className="space-y-2 text-xs font-mono">
            {([
              ['Elapsed time', formatElapsedTime(tech.metrics.elapsed_time)],
              ['Prompt tokens', formatWholeNumber(tech.metrics.prompt_tokens)],
              ['Completion tokens', formatWholeNumber(tech.metrics.completion_tokens)],
              ['Total tokens', formatWholeNumber(tech.metrics.total_tokens)],
              ['Word count', formatWholeNumber(tech.wordCount)],
            ] as const).map(([label, val]) => (
              <div key={label} className="flex items-center justify-between">
                <span style={{ color: 'var(--text-subtle)' }}>{label}</span>
                <span>{val}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  /* Whether we're in "results" mode (sidebar + right panel) */
  const hasStarted = loading || !!result || !!baselineResult

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
                {formatScorePercent(result.best_result?.scores?.overall)}
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
              {/* Top row: Mode + Subject + Difficulty */}
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                  <label htmlFor="mode-landing" className="block text-sm font-medium mb-1.5">
                    Run Mode
                  </label>
                  <select
                    id="mode-landing"
                    value={runMode}
                    onChange={(e) => setRunMode(e.target.value as RunMode)}
                    className="w-full px-3 py-2 rounded-md text-sm outline-none"
                    style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--surface)' }}
                  >
                    <option value="normal">Normal mode</option>
                    <option value="benchmark">Benchmark mode</option>
                    <option value="baseline">Baseline mode</option>
                  </select>
                </div>
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
                    {['basic', 'intermediate', 'advanced'].map((d) => (
                      <option key={d} value={d}>{d.charAt(0).toUpperCase() + d.slice(1)}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Available Few-Shot Concepts (opens modal) — benchmark + normal */}
              {canManageFewShotExamples && Object.keys(exampleTypes).length > 0 && (
                <div className="mb-4 flex gap-2">
                  <button
                    type="button"
                    onClick={() => setShowExampleTypes(true)}
                    className="flex items-center gap-1.5 text-xs font-medium px-2 py-1 rounded"
                    style={{ color: 'var(--text-subtle)', background: 'var(--bg)', border: '1px solid var(--border)' }}
                  >
                    ▶ Available Few-Shot Concepts
                  </button>
                  <button
                    type="button"
                    onClick={() => { setShowAddExample(true); setAddExStatus(''); setAddExAnalyzed(false); setAddExAnalyzed2(false); setAddExProblem2(''); setAddExSolution2('') }}
                    className="flex items-center gap-1.5 text-xs font-medium px-2 py-1 rounded"
                    style={{ color: '#fff', background: 'var(--accent)', border: 'none' }}
                  >
                    + Add Example
                  </button>
                </div>
              )}

              {(runMode === 'benchmark' || runMode === 'baseline') && (
                <div className="mb-4 p-3 rounded-md" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
                  <label htmlFor="ground-truth-landing" className="block text-sm font-medium mb-1.5">
                    Expected Answer {runMode === 'benchmark' ? '(required)' : '(optional)'}
                  </label>
                  <input
                    id="ground-truth-landing"
                    value={groundTruth}
                    onChange={(e) => setGroundTruth(e.target.value)}
                    placeholder="e.g., x = 4 or x = 5"
                    className="w-full px-3 py-2 rounded-md text-sm font-mono outline-none"
                    style={{
                      border: `1px solid ${runMode === 'benchmark' && !groundTruth.trim() ? '#ef4444' : 'var(--border)'}`,
                      color: 'var(--text)',
                      background: 'var(--surface)',
                    }}
                  />
                  <p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                    {runMode === 'benchmark'
                      ? 'Benchmark mode enforces ground-truth scoring for thesis-grade comparison.'
                      : 'Providing an expected answer enables accurate scoring instead of heuristics.'}
                  </p>
                </div>
              )}

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
                  {modeLabel}: {modeHint}
                </span>
                {runMode === 'benchmark' || runMode === 'baseline' ? (
                  <button
                    type="submit"
                    disabled={isDisabled}
                    className="flex items-center rounded-md text-sm font-medium transition-colors overflow-hidden"
                    style={{
                      background: isDisabled ? 'var(--border-strong)' : runMode === 'baseline' ? '#57534e' : 'var(--accent)',
                      color: isDisabled ? 'var(--text-muted)' : '#fff',
                      cursor: isDisabled ? 'not-allowed' : 'pointer',
                      padding: 0,
                    }}
                  >
                    <span className="px-5 py-2.5">{runMode === 'baseline' ? 'Run Baseline' : 'Run Benchmark'}</span>
                    <span
                      role="button"
                      onClick={(e) => { e.preventDefault(); e.stopPropagation(); setBenchmarkRuns(benchmarkRuns === 1 ? 3 : 1); }}
                      className="px-3 py-2.5 text-xs font-mono transition-colors"
                      style={{
                        borderLeft: '1px solid rgba(255,255,255,0.18)',
                        color: 'rgba(255,255,255,0.75)',
                      }}
                    >
                      &times;{benchmarkRuns}
                    </span>
                  </button>
                ) : (
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
                    Run Normal Mode &rarr;
                  </button>
                )}
              </div>
            </form>
          </div>

          {renderErrorPanel()}
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
              {/* Mode */}
              <div>
                <label htmlFor="run-mode" className="block text-sm font-medium mb-1.5">
                  Run Mode
                </label>
                <select
                  id="run-mode"
                  value={runMode}
                  onChange={(e) => setRunMode(e.target.value as RunMode)}
                  disabled={loading}
                  className="w-full px-3 py-2 rounded-md text-sm outline-none transition disabled:cursor-not-allowed"
                  style={{
                    border: '1px solid var(--border)',
                    color: 'var(--text)',
                    background: loading ? 'var(--bg)' : 'var(--surface)',
                  }}
                >
                  <option value="normal">Normal</option>
                  <option value="benchmark">Benchmark</option>
                  <option value="baseline">Baseline</option>
                </select>
              </div>

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
                  {['basic', 'intermediate', 'advanced'].map((d) => (
                    <option key={d} value={d}>{d.charAt(0).toUpperCase() + d.slice(1)}</option>
                  ))}
                </select>
              </div>

              {/* Available Few-Shot Concepts (opens modal) — benchmark + normal */}
              {canManageFewShotExamples && Object.keys(exampleTypes).length > 0 && (
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setShowExampleTypes(true)}
                    className="flex items-center gap-1.5 text-xs font-medium px-2 py-1 rounded"
                    style={{ color: 'var(--text-subtle)', background: 'var(--bg)', border: '1px solid var(--border)' }}
                  >
                    ▶ Available Few-Shot Concepts
                  </button>
                  <button
                    type="button"
                    onClick={() => { setShowAddExample(true); setAddExStatus(''); setAddExAnalyzed(false); setAddExAnalyzed2(false); setAddExProblem2(''); setAddExSolution2('') }}
                    className="flex items-center gap-1.5 text-xs font-medium px-2 py-1 rounded"
                    style={{ color: '#fff', background: 'var(--accent)', border: 'none' }}
                  >
                    + Add Example
                  </button>
                </div>
              )}

              {(runMode === 'benchmark' || runMode === 'baseline') && (
                <div className="p-3 rounded-md" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
                  <label htmlFor="ground-truth" className="block text-sm font-medium mb-1.5">
                    Expected Answer {runMode === 'benchmark' ? '(required)' : '(optional)'}
                  </label>
                  <input
                    id="ground-truth"
                    value={groundTruth}
                    onChange={(e) => setGroundTruth(e.target.value)}
                    disabled={loading}
                    placeholder="e.g., x = 4 or x = 5"
                    className="w-full px-3 py-2 rounded-md text-sm font-mono outline-none transition"
                    style={{
                      border: `1px solid ${runMode === 'benchmark' && !groundTruth.trim() ? '#ef4444' : 'var(--border)'}`,
                      color: 'var(--text)',
                      background: loading ? 'var(--bg)' : 'var(--surface)',
                    }}
                  />
                  <p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                    {runMode === 'benchmark'
                      ? 'Required for benchmark mode and thesis-grade scoring.'
                      : 'Providing an expected answer enables accurate scoring.'}
                  </p>
                </div>
              )}

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
              {runMode === 'benchmark' || runMode === 'baseline' ? (
                <button
                  type="submit"
                  disabled={isDisabled}
                  className="flex items-center w-full rounded-md text-sm font-medium transition-colors overflow-hidden"
                  style={{
                    background: isDisabled ? 'var(--border-strong)' : runMode === 'baseline' ? '#57534e' : 'var(--accent)',
                    color: isDisabled ? 'var(--text-muted)' : '#fff',
                    cursor: isDisabled ? 'not-allowed' : 'pointer',
                    padding: 0,
                  }}
                >
                  <span className="flex-1 py-2.5">{loading ? 'Running...' : runMode === 'baseline' ? 'Run Baseline' : 'Run Benchmark'}</span>
                  <span
                    role="button"
                    onClick={(e) => { e.preventDefault(); e.stopPropagation(); setBenchmarkRuns(benchmarkRuns === 1 ? 3 : 1); }}
                    className="px-3 py-2.5 text-xs font-mono transition-colors"
                    style={{
                      borderLeft: '1px solid rgba(255,255,255,0.18)',
                      color: 'rgba(255,255,255,0.75)',
                    }}
                  >
                    &times;{benchmarkRuns}
                  </span>
                </button>
              ) : (
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
                  {loading ? 'Running...' : 'Run'}
                </button>
              )}
            </form>

            {renderErrorPanel()}
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
              <p className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>
                This may take a few seconds
              </p>
            </div>
          )}

          {/* ═══ Baseline Results ═══ */}
          {baselineResult && (
            <div className="space-y-6">
              {/* Header */}
              <div
                className="rounded-lg overflow-hidden"
                style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
              >
                <div
                  className="flex items-center justify-between px-6 py-4"
                  style={{ borderBottom: '1px solid var(--border)' }}
                >
                  <div className="flex items-center gap-3">
                    <h2 className="text-lg font-semibold">Raw Baseline</h2>
                    <span
                      className="font-mono text-xs px-2.5 py-1 rounded"
                      style={{ background: '#57534e', color: '#fff' }}
                    >
                      NO TECHNIQUE
                    </span>
                    <span
                      className="font-mono text-xs px-2.5 py-1 rounded"
                      style={{ background: 'var(--bg)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                    >
                      {baselineResult.model_name}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
                      {baselineResult.runs_succeeded}/{baselineResult.runs_requested} run{baselineResult.runs_requested !== 1 ? 's' : ''} succeeded
                    </span>
                    {baselineResult.runs_requested >= 3 && (
                      <button
                        onClick={handleSaveBaselineToDb}
                        disabled={savingBaselineToDb || didSaveBaselineToDb}
                        className="px-3 py-1.5 rounded-md text-xs font-medium transition-colors disabled:cursor-not-allowed"
                        style={{
                          border: '1px solid var(--border)',
                          color: didSaveBaselineToDb ? 'var(--green)' : 'var(--text)',
                          background: 'var(--surface)',
                        }}
                      >
                        {savingBaselineToDb ? 'Saving\u2026' : didSaveBaselineToDb ? '\u2713 Saved' : 'Save to DB'}
                      </button>
                    )}
                  </div>
                </div>

                {/* Baseline save status */}
                {baselineSaveStatus && (() => {
                  const isSuccess = baselineSaveStatus.startsWith('Saved')
                  return (
                    <div
                      className="px-6 py-2 text-xs font-mono"
                      style={{
                        background: isSuccess ? '#f0fdf4' : '#fef2f2',
                        borderBottom: '1px solid var(--border)',
                        color: isSuccess ? 'var(--green)' : '#dc2626',
                      }}
                    >
                      {baselineSaveStatus}
                    </div>
                  )
                })()}



                <div className="p-6 space-y-6">
                  {/* Prompt Used */}
                  <details>
                    <summary
                      className="text-[11px] font-mono uppercase tracking-wider cursor-pointer select-none hover:underline"
                      style={{ color: 'var(--text-subtle)' }}
                    >
                      Raw Prompt Sent
                    </summary>
                    <div
                      className="mt-2 p-4 rounded-md overflow-auto max-h-72"
                      style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
                    >
                      <pre className="text-sm font-mono whitespace-pre-wrap leading-relaxed">
                        {baselineResult.prompt_used}
                      </pre>
                    </div>
                  </details>

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
                      style={{ background: '#f5f5f4', border: '1px solid #d6d3d1' }}
                    >
                      <pre className="text-sm font-mono whitespace-pre-wrap leading-relaxed">
                        {baselineResult.best_response || 'No response'}
                      </pre>
                    </div>
                  </div>

                  {/* Performance Scores */}
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <p
                        className="text-[11px] font-mono uppercase tracking-wider"
                        style={{ color: 'var(--text-subtle)' }}
                      >
                        Performance Scores
                      </p>
                      <div
                        className="flex items-center rounded-md overflow-hidden text-[10px] font-mono"
                        style={{ border: '1px solid var(--border)' }}
                      >
                        <button
                          type="button"
                          onClick={() => setPerfScoreFormat('percent')}
                          className="px-2.5 py-1 transition-colors"
                          style={{
                            background: perfScoreFormat === 'percent' ? '#57534e' : 'var(--surface)',
                            color: perfScoreFormat === 'percent' ? '#fff' : 'var(--text-muted)',
                          }}
                        >
                          %
                        </button>
                        <button
                          type="button"
                          onClick={() => setPerfScoreFormat('decimal')}
                          className="px-2.5 py-1 transition-colors"
                          style={{
                            background: perfScoreFormat === 'decimal' ? '#57534e' : 'var(--surface)',
                            color: perfScoreFormat === 'decimal' ? '#fff' : 'var(--text-muted)',
                          }}
                        >
                          0.00
                        </button>
                      </div>
                    </div>

                    {!baselineResult.ground_truth_used && (
                      <div
                        className="mb-3 px-3 py-2 rounded-md text-xs"
                        style={{ background: '#fffbeb', border: '1px solid #fde68a', color: 'var(--amber)' }}
                      >
                        Accuracy is heuristic — no expected answer was provided.
                      </div>
                    )}

                    <div className="grid grid-cols-4 gap-3">
                      {[
                        {
                          label: !baselineResult.ground_truth_used ? 'ACCURACY*' : 'ACCURACY',
                          value: baselineResult.scores.accuracy,
                        },
                        {
                          label: baselineResult.scores.consistency_is_provisional
                            ? 'CONSISTENCY*'
                            : 'CONSISTENCY',
                          value: baselineResult.scores.consistency,
                        },
                      ].map((s) => (
                        <div
                          key={s.label}
                          className="p-3 rounded-md text-center"
                          style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
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
                            {formatScore(s.value, perfScoreFormat, 'PROV')}
                          </p>
                        </div>
                      ))}
                      {/* Efficiency card with info button */}
                      <div
                        className="p-3 rounded-md text-center"
                        style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
                      >
                        <p
                          className="text-[9px] font-mono uppercase tracking-widest mb-1"
                          style={{ color: 'var(--text-subtle)' }}
                        >
                          EFFICIENCY
                        </p>
                        <div className="flex items-center justify-center gap-1.5">
                          <p
                            className="text-2xl font-mono font-light"
                            style={{ color: scoreColor(baselineResult.scores.efficiency ?? 0) }}
                          >
                            {formatScore(baselineResult.scores.efficiency, perfScoreFormat)}
                          </p>
                          <button
                            type="button"
                            onClick={(event) => {
                              if (efficiencyInfo?.technique === 'baseline') {
                                setEfficiencyInfo(null)
                              } else {
                                const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
                                setEfficiencyInfo({ technique: 'baseline', x: rect.right + 8, y: rect.top - 4 })
                              }
                            }}
                            className="inline-flex h-[18px] w-[18px] items-center justify-center rounded-full text-[9px] font-semibold leading-none transition-colors shrink-0"
                            style={{
                              background: efficiencyInfo?.technique === 'baseline' ? 'var(--accent)' : 'transparent',
                              color: efficiencyInfo?.technique === 'baseline' ? '#fff' : 'var(--text-subtle)',
                              border: efficiencyInfo?.technique === 'baseline' ? '1px solid var(--accent)' : '1px solid var(--border)',
                            }}
                            title="Efficiency breakdown"
                          >
                            i
                          </button>
                        </div>
                      </div>
                      {/* Overall (avg) card */}
                      <div
                        className="p-3 rounded-md text-center"
                        style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
                      >
                        <p
                          className="text-[9px] font-mono uppercase tracking-widest mb-1"
                          style={{ color: 'var(--text-subtle)' }}
                        >
                          AVG
                        </p>
                        <p
                          className="text-2xl font-mono font-light"
                          style={{ color: scoreColor(baselineResult.scores.overall ?? 0) }}
                        >
                          {formatScore(baselineResult.scores.overall, perfScoreFormat)}
                        </p>
                      </div>
                    </div>

                    {/* Efficiency popover for baseline */}
                    {efficiencyInfo?.technique === 'baseline' && (
                      <div className="fixed inset-0 z-50" onClick={() => setEfficiencyInfo(null)}>
                        <div
                          className="fixed w-60 rounded-lg p-4 shadow-xl"
                          style={{
                            left: Math.min(efficiencyInfo.x, (typeof window !== 'undefined' ? window.innerWidth : 1200) - 252),
                            top: Math.min(Math.max(efficiencyInfo.y, 8), (typeof window !== 'undefined' ? window.innerHeight : 800) - 222),
                            background: 'var(--surface)',
                            border: '1px solid var(--border)',
                          }}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <p className="text-[10px] font-mono uppercase tracking-widest mb-3" style={{ color: 'var(--text-subtle)' }}>
                            Efficiency Breakdown
                          </p>
                          <div className="space-y-2 text-xs font-mono">
                            <div className="flex justify-between"><span style={{ color: 'var(--text-muted)' }}>Time</span><span>{formatElapsedTime(baselineResult.metrics.elapsed_time)}</span></div>
                            <div className="flex justify-between"><span style={{ color: 'var(--text-muted)' }}>Total tokens</span><span>{formatWholeNumber(baselineResult.metrics.total_tokens)}</span></div>
                            <div className="flex justify-between"><span style={{ color: 'var(--text-muted)' }}>Prompt tokens</span><span>{formatWholeNumber(baselineResult.metrics.prompt_tokens)}</span></div>
                            <div className="flex justify-between"><span style={{ color: 'var(--text-muted)' }}>Completion tokens</span><span>{formatWholeNumber(baselineResult.metrics.completion_tokens)}</span></div>
                            <div className="flex justify-between"><span style={{ color: 'var(--text-muted)' }}>Words</span><span>{countWords(baselineResult.best_response)}</span></div>
                          </div>
                        </div>
                      </div>
                    )}

                    {!baselineResult.ground_truth_used && (
                      <p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                        * Heuristic accuracy is directionally useful but less reliable than ground-truth scoring.
                      </p>
                    )}
                  </div>

                  {/* Individual Runs button (if > 1) */}
                  {baselineResult.run_history.length > 1 && (
                    <button
                      type="button"
                      onClick={() => setShowIndividualRuns(true)}
                      className="text-[11px] font-mono uppercase tracking-wider cursor-pointer select-none hover:underline"
                      style={{ color: 'var(--text-subtle)', background: 'none', border: 'none', padding: 0 }}
                    >
                      &#9660; Individual Runs ({baselineResult.run_history.length})
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ═══ Results ═══ */}
          {result && (
            <>
              {/* NORMAL MODE: Simplified View */}
              {activeMode === 'normal' && (
                <div className="space-y-4">
                  {/* Header row: technique + tier + actions */}
                  <div
                    className="rounded-lg overflow-hidden"
                    style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
                  >
                    <div
                      className="flex items-center justify-between px-6 py-3"
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
                        {result.selection_source && (() => {
                          const tierInfo = getTierInfo(result.selection_source as SelectionSource)
                          return (
                            <span
                              className="font-mono text-xs px-2.5 py-1 rounded"
                              style={{ background: tierInfo.textColor, color: tierInfo.bgColor }}
                            >
                              {tierInfo.tierName}
                            </span>
                          )
                        })()}
                      </div>

                    </div>

                    {/* Tier one-liner */}
                    {result.selection_source && (() => {
                      const tierInfo = getTierInfo(result.selection_source as SelectionSource)
                      const details = result.selection_details as Record<string, any> || {}
                      const summary = buildTierSummary(
                        result.selection_source as SelectionSource,
                        details,
                        result.best_technique,
                      )
                      return (
                        <div
                          className="px-6 py-2 text-xs"
                          style={{ background: tierInfo.bgColor, borderBottom: `1px solid ${tierInfo.borderColor}`, color: 'var(--text-muted)' }}
                        >
                          {summary}
                        </div>
                      )
                    })()}

                    <div className="p-6 space-y-5">
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
                            {getDisplayPrompt(result.best_result?.prompt)}
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

                  {/* Historical Performance + Coverage — shown when Tier 1/2 selected */}
                  {result.selection_source !== 'runtime_scores' && (() => {
                    const histStats = getHistoricalStats(
                      result.selection_source as SelectionSource,
                      result.selection_details as Record<string, any>,
                    )
                    return (
                  <div
                    className="rounded-lg overflow-hidden"
                    style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
                  >
                    <div className="p-6">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <p
                            className="text-[11px] font-mono uppercase tracking-wider"
                            style={{ color: 'var(--text-subtle)' }}
                          >
                            Historical Performance
                          </p>
                          <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ background: 'var(--bg)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                            {subject} &middot; {difficulty}
                          </span>
                        </div>
                        {coverageData && coverageData.problem_count > 0 && (
                          <div className="flex items-center gap-3 text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                            <span>{coverageData.problem_count} benchmarked</span>
                            {coverageData.ground_truth_count > 0 && (
                              <span style={{ color: 'var(--green)' }}>{coverageData.ground_truth_count} with ground truth</span>
                            )}
                          </div>
                        )}
                      </div>
                          <div className="grid grid-cols-4 gap-3">
                            {[
                              {
                                label: 'WIN RATE',
                                display: histStats.winRate != null
                                  ? `${(histStats.winRate * 100).toFixed(0)}%`
                                  : 'N/A',
                                raw: histStats.winRate,
                              },
                              {
                                label: 'AVG SCORE',
                                display: histStats.avgScore != null
                                  ? `${(histStats.avgScore * 100).toFixed(1)}%`
                                  : 'N/A',
                                raw: histStats.avgScore,
                              },
                              {
                                label: 'DATA POINTS',
                                display: `${histStats.samples}`,
                                raw: histStats.samples > 0 ? histStats.samples / 20 : null,
                              },
                              {
                                label: 'LEAD',
                                display: histStats.lead != null
                                  ? `+${(histStats.lead * 100).toFixed(1)}%`
                                  : 'N/A',
                                raw: histStats.lead,
                              },
                            ].map((s) => (
                              <div
                                key={s.label}
                                className="p-3 rounded-md text-center"
                                style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
                              >
                                <p
                                  className="text-[9px] font-mono uppercase tracking-widest mb-1"
                                  style={{ color: 'var(--text-subtle)' }}
                                >
                                  {s.label}
                                </p>
                                <p
                                  className="text-2xl font-mono font-light"
                                  style={{ color: s.raw != null ? scoreColor(s.raw) : 'var(--text-muted)' }}
                                >
                                  {s.display}
                                </p>
                              </div>
                            ))}
                          </div>
                          {histStats.ranking.length > 1 && (
                            <div className="mt-3 flex items-center gap-4 text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                              {histStats.ranking.map((r, i) => (
                                <span key={r.technique} style={{ color: i === 0 ? 'var(--green)' : 'var(--text-muted)' }}>
                                  {i === 0 ? '\u2605 ' : ''}{r.technique}: {(r.avgScore * 100).toFixed(1)}% avg
                                </span>
                              ))}
                            </div>
                          )}
                    </div>
                  </div>
                    )
                  })()}

                  {/* Coverage Indicator — only when runtime selection (no historical panel above) */}
                  {result.selection_source === 'runtime_scores' && coverageData && coverageData.problem_count > 0 && (
                    <div
                      className="rounded-lg overflow-hidden px-5 py-3"
                      style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-[11px] font-mono uppercase tracking-wider" style={{ color: 'var(--text-subtle)' }}>
                            Benchmark Coverage
                          </span>
                          <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ background: 'var(--bg)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                            {subject} &middot; {difficulty}
                          </span>
                        </div>
                        <div className="flex items-center gap-3 text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                          <span>{coverageData.problem_count} problem{coverageData.problem_count !== 1 ? 's' : ''} benchmarked</span>
                          {coverageData.ground_truth_count > 0 && (
                            <span style={{ color: 'var(--green)' }}>{coverageData.ground_truth_count} with ground truth</span>
                          )}
                          {coverageData.total_with_winner > 0 && coverageData.win_counts && (
                            <span>
                              {Object.entries(coverageData.win_counts).map(([tech, w]) => (
                                <span key={tech} className="ml-2">
                                  {tech}: {Math.round((w / coverageData.total_with_winner) * 100)}% wins
                                </span>
                              ))}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Technique Comparison — only when runtime selection (no DB data) */}
                  {result.selection_source === 'runtime_scores' && (
                    <div
                      className="rounded-lg overflow-hidden"
                      style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
                    >
                        <div
                          className="flex items-center justify-between px-6 py-4"
                          style={{ borderBottom: '1px solid var(--border)' }}
                        >
                          <h2 className="text-lg font-semibold">Technique Comparison</h2>
                          <div className="flex items-center gap-3">
                            <span className="font-mono text-xs" style={{ color: 'var(--text-subtle)' }}>
                              {attemptedTechniqueCount} techniques - {successfulTechniqueCount} successful
                            </span>
                            <div
                              className="flex items-center rounded-md overflow-hidden text-[10px] font-mono"
                              style={{ border: '1px solid var(--border)' }}
                            >
                              <button
                                type="button"
                                onClick={() => setCompScoreFormat('percent')}
                                className="px-2.5 py-1 transition-colors"
                                style={{
                                  background: compScoreFormat === 'percent' ? 'var(--accent)' : 'var(--surface)',
                                  color: compScoreFormat === 'percent' ? '#fff' : 'var(--text-muted)',
                                }}
                              >
                                %
                              </button>
                              <button
                                type="button"
                                onClick={() => setCompScoreFormat('decimal')}
                                className="px-2.5 py-1 transition-colors"
                                style={{
                                  background: compScoreFormat === 'decimal' ? 'var(--accent)' : 'var(--surface)',
                                  color: compScoreFormat === 'decimal' ? '#fff' : 'var(--text-muted)',
                                }}
                              >
                                0.00
                              </button>
                            </div>
                          </div>
                        </div>
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead>
                              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                {['Technique', 'Accuracy', 'Consistency', 'Efficiency', 'Overall', 'Details'].map(
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
                              {techniqueRows.map((tech) => {
                                const isBest = tech.technique === result.best_technique
                                return (
                                  <tr
                                    key={tech.technique}
                                    className={isBest ? 'font-medium' : ''}
                                    style={{
                                      borderBottom: '1px solid var(--border)',
                                      background: isBest ? '#f0fdf4' : undefined,
                                      boxShadow: isBest ? 'inset 3px 0 0 var(--green)' : undefined,
                                    }}
                                  >
                                    <td className="px-6 py-3 font-medium">
                                      {tech.technique?.toUpperCase() ?? 'N/A'}
                                      {isBest && (
                                        <span
                                          className="ml-2 text-[10px] font-mono px-1.5 py-0.5 rounded"
                                          style={{ background: '#dcfce7', color: 'var(--green)' }}
                                        >
                                          best
                                        </span>
                                      )}
                                      {!tech.success && (
                                        <span
                                          className="ml-2 text-[10px] font-mono px-1.5 py-0.5 rounded"
                                          style={{ background: '#fef2f2', color: '#b91c1c', border: '1px solid #fecaca' }}
                                        >
                                          failed
                                        </span>
                                      )}
                                    </td>
                                    <td className="text-center px-4 py-3 font-mono">
                                      {formatScore(tech.accuracy, compScoreFormat)}
                                    </td>
                                    <td className="text-center px-4 py-3 font-mono">
                                      {tech.consistencyAvailable ? formatScore(tech.consistency, compScoreFormat, 'PROV') : 'PROV'}
                                    </td>
                                    <td className="text-center px-4 py-3">
                                      {renderEfficiencyCell(tech)}
                                    </td>
                                    <td
                                      className="text-center px-4 py-3 font-mono font-semibold"
                                      style={{ color: scoreColor(tech.overall) }}
                                    >
                                      {formatScore(tech.overall, compScoreFormat)}
                                      {tech.overallIsProvisional ? '*' : ''}
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
                  )}
                </div>
              )}

              {/* BENCHMARK MODE: Full Detail View */}
              {activeMode === 'benchmark' && (
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
                    {result.selection_source && (() => {
                      const tierInfo = getTierInfo(result.selection_source as SelectionSource)
                      return (
                        <span
                          className="font-mono text-xs px-2.5 py-1 rounded"
                          style={{ background: tierInfo.textColor, color: tierInfo.bgColor }}
                        >
                          {tierInfo.tierName}
                        </span>
                      )
                    })()}
                  </div>
                  {benchmarkRuns > 1 && !result.few_shot_unavailable && (
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
                  )}
                </div>

                {/* Save / Export status */}
                {benchmarkRuns > 1 && saveStatus && (() => {
                  const isSuccess =
                    saveStatus.startsWith('Saved') ||
                    saveStatus.startsWith('Exported as JSON and saved') ||
                    saveStatus.startsWith('Consistency test complete')
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

                {/* Few-shot unavailable warning */}
                {result.few_shot_unavailable && (
                  <div
                    className="px-6 py-3 text-xs"
                    style={{ background: '#fffbeb', borderBottom: '1px solid #fde68a', color: 'var(--amber)' }}
                  >
                    {'\u26a0'} Few-shot was skipped: {result.few_shot_error || 'No matching examples found in example_problems.json.'}
                    {' '}Only zero-shot results are shown. Save and Export are disabled.
                  </div>
                )}

                {/* ── Selection Strategy (one-line summary) ── */}
                {result.selection_source && result.selection_source !== 'runtime_scores' && (() => {
                  const tierInfo = getTierInfo(result.selection_source as SelectionSource)
                  const details = result.selection_details as Record<string, any> || {}
                  const summary = buildTierSummary(
                    result.selection_source as SelectionSource,
                    details,
                    result.best_technique,
                  )

                  return (
                    <div
                      className="px-6 py-2.5 border-b flex items-center gap-2 text-xs"
                      style={{ background: tierInfo.bgColor, borderColor: tierInfo.borderColor }}
                    >
                      <span
                        className="font-mono font-bold px-2 py-0.5 rounded shrink-0"
                        style={{ background: tierInfo.textColor, color: tierInfo.bgColor }}
                      >
                        {tierInfo.tierName}
                      </span>
                      <span className="font-medium" style={{ color: tierInfo.textColor }}>
                        {tierInfo.strategyName}
                      </span>
                      <span style={{ color: 'var(--text-muted)' }}>·</span>
                      <span style={{ color: 'var(--text-muted)' }}>
                        {summary}
                      </span>
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
                  {/* Prompt Used (collapsible) */}
                  <details>
                    <summary
                      className="text-[11px] font-mono uppercase tracking-wider cursor-pointer select-none hover:underline"
                      style={{ color: 'var(--text-subtle)' }}
                    >
                      Prompt Used
                    </summary>
                    <div
                      className="mt-2 p-4 rounded-md overflow-auto max-h-72"
                      style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
                    >
                      <pre className="text-sm font-mono whitespace-pre-wrap leading-relaxed">
                        {getDisplayPrompt(result.best_result?.prompt)}
                      </pre>
                    </div>
                  </details>

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

                  {/* Performance Scores (inline) */}
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <p
                        className="text-[11px] font-mono uppercase tracking-wider"
                        style={{ color: 'var(--text-subtle)' }}
                      >
                        Performance Scores
                      </p>
                      <div
                        className="flex items-center rounded-md overflow-hidden text-[10px] font-mono"
                        style={{ border: '1px solid var(--border)' }}
                      >
                        <button
                          type="button"
                          onClick={() => setPerfScoreFormat('percent')}
                          className="px-2.5 py-1 transition-colors"
                          style={{
                            background: perfScoreFormat === 'percent' ? 'var(--accent)' : 'var(--surface)',
                            color: perfScoreFormat === 'percent' ? '#fff' : 'var(--text-muted)',
                          }}
                        >
                          %
                        </button>
                        <button
                          type="button"
                          onClick={() => setPerfScoreFormat('decimal')}
                          className="px-2.5 py-1 transition-colors"
                          style={{
                            background: perfScoreFormat === 'decimal' ? 'var(--accent)' : 'var(--surface)',
                            color: perfScoreFormat === 'decimal' ? '#fff' : 'var(--text-muted)',
                          }}
                        >
                          0.00
                        </button>
                      </div>
                    </div>

                    {!runUsedGroundTruth && (
                      <div
                        className="mb-3 px-3 py-2 rounded-md text-xs"
                        style={{ background: '#fffbeb', border: '1px solid #fde68a', color: 'var(--amber)' }}
                      >
                        Accuracy is heuristic — no expected answer was provided.
                      </div>
                    )}

                    <div className="grid grid-cols-4 gap-3">
                      {[
                        { label: 'OVERALL', value: result.best_result?.scores?.overall },
                        {
                          label: !runUsedGroundTruth ? 'ACCURACY*' : 'ACCURACY',
                          value: result.best_result?.scores?.accuracy,
                        },
                        {
                          label: result.best_result?.scores?.consistency_is_provisional
                            ? 'CONSISTENCY*'
                            : 'CONSISTENCY',
                          value: result.best_result?.scores?.consistency,
                        },
                        { label: 'EFFICIENCY', value: result.best_result?.scores?.efficiency },
                      ].map((s) => (
                        <div
                          key={s.label}
                          className="p-3 rounded-md text-center"
                          style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
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
                            {formatScore(s.value, perfScoreFormat, 'PROV')}
                          </p>
                        </div>
                      ))}
                    </div>

                    {!runUsedGroundTruth && (
                      <p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                        * Heuristic accuracy is directionally useful but less reliable than ground-truth scoring.
                      </p>
                    )}
                  </div>
                </div>
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
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-xs" style={{ color: 'var(--text-subtle)' }}>
                      {attemptedTechniqueCount} techniques - {successfulTechniqueCount} successful
                    </span>
                    <div
                      className="flex items-center rounded-md overflow-hidden text-[10px] font-mono"
                      style={{ border: '1px solid var(--border)' }}
                    >
                      <button
                        type="button"
                        onClick={() => setCompScoreFormat('percent')}
                        className="px-2.5 py-1 transition-colors"
                        style={{
                          background: compScoreFormat === 'percent' ? 'var(--accent)' : 'var(--surface)',
                          color: compScoreFormat === 'percent' ? '#fff' : 'var(--text-muted)',
                        }}
                      >
                        %
                      </button>
                      <button
                        type="button"
                        onClick={() => setCompScoreFormat('decimal')}
                        className="px-2.5 py-1 transition-colors"
                        style={{
                          background: compScoreFormat === 'decimal' ? 'var(--accent)' : 'var(--surface)',
                          color: compScoreFormat === 'decimal' ? '#fff' : 'var(--text-muted)',
                        }}
                      >
                        0.00
                      </button>
                    </div>
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border)' }}>
                        {['Technique', 'Accuracy', 'Consistency', 'Efficiency', 'Overall', 'Details'].map(
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
                      {techniqueRows.map((tech) => {
                        const isBest = tech.technique === result.best_technique
                        return (
                          <tr
                            key={tech.technique}
                            className={isBest ? 'font-medium' : ''}
                            style={{
                              borderBottom: '1px solid var(--border)',
                              background: isBest ? '#f0fdf4' : undefined,
                              boxShadow: isBest ? 'inset 3px 0 0 var(--green)' : undefined,
                            }}
                          >
                            <td className="px-6 py-3 font-medium">
                              {tech.technique?.toUpperCase() ?? 'N/A'}
                              {isBest && (
                                <span
                                  className="ml-2 text-[10px] font-mono px-1.5 py-0.5 rounded"
                                  style={{ background: '#dcfce7', color: 'var(--green)' }}
                                >
                                  best
                                </span>
                              )}
                              {!tech.success && (
                                <span
                                  className="ml-2 text-[10px] font-mono px-1.5 py-0.5 rounded"
                                  style={{ background: '#fef2f2', color: '#b91c1c', border: '1px solid #fecaca' }}
                                >
                                  failed
                                </span>
                              )}
                            </td>
                            <td className="text-center px-4 py-3 font-mono">
                              {formatScore(tech.accuracy, compScoreFormat)}
                            </td>
                            <td className="text-center px-4 py-3 font-mono">
                              {tech.consistencyAvailable ? formatScore(tech.consistency, compScoreFormat, 'PROV') : 'PROV'}
                            </td>
                            <td className="text-center px-4 py-3">
                              {renderEfficiencyCell(tech)}
                            </td>
                            <td
                              className="text-center px-4 py-3 font-mono font-semibold"
                              style={{ color: scoreColor(tech.overall) }}
                            >
                              {formatScore(tech.overall, compScoreFormat)}
                              {tech.overallIsProvisional ? '*' : ''}
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
            </>
          )}
        </section>
        </div>
      )}
      {/* ═══ Efficiency Breakdown Popover ═══ */}
      {renderEfficiencyPopover()}
      {/* ═══ Baseline Individual Runs Modal ═══ */}
      {showIndividualRuns && baselineResult && baselineResult.run_history.length > 1 && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ background: 'rgba(0,0,0,0.25)' }}
          onClick={() => setShowIndividualRuns(false)}
        >
          <div
            className="relative w-full max-w-3xl mx-4 rounded-lg shadow-xl overflow-hidden flex flex-col"
            style={{ background: 'var(--surface)', maxHeight: '85vh' }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div
              className="flex items-center justify-between px-6 py-4 shrink-0"
              style={{ borderBottom: '1px solid var(--border)' }}
            >
              <div className="flex items-center gap-3">
                <h2 className="text-lg font-semibold">Individual Runs</h2>
                <span
                  className="font-mono text-xs px-2 py-1 rounded"
                  style={{ color: 'var(--text)', border: '1px solid var(--border)' }}
                >
                  {baselineResult.run_history.length} RUNS
                </span>
              </div>
              <button
                onClick={() => setShowIndividualRuns(false)}
                className="w-8 h-8 flex items-center justify-center rounded-md transition-colors"
                style={{ color: 'var(--text-muted)' }}
              >
                &#x2715;
              </button>
            </div>

            {/* Modal body — scrollable */}
            <div className="overflow-y-auto px-6 py-4 space-y-4">
              {baselineResult.run_history.map((run) => (
                <div
                  key={run.run_index}
                  className="rounded-lg overflow-hidden"
                  style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
                >
                  {/* Run header */}
                  <div
                    className="px-4 py-2 flex items-center justify-between"
                    style={{ borderBottom: '1px solid var(--border)' }}
                  >
                    <span className="text-xs font-mono font-semibold" style={{ color: 'var(--text)' }}>
                      Run #{run.run_index + 1}{!run.success && ' — Failed'}
                    </span>
                    {run.success && (
                      <div className="flex gap-4 text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                        <span>ACC <span style={{ color: scoreColor(run.scores.accuracy) }}>{formatScore(run.scores.accuracy, perfScoreFormat)}</span></span>
                        <span>CON <span>{formatScore(run.scores.consistency, perfScoreFormat, 'N/A')}</span></span>
                        <span>EFF <span style={{ color: scoreColor(run.scores.efficiency) }}>{formatScore(run.scores.efficiency, perfScoreFormat)}</span></span>
                        <span>AVG <span style={{ color: scoreColor(run.scores.overall) }}>{formatScore(run.scores.overall, perfScoreFormat)}</span></span>
                      </div>
                    )}
                  </div>

                  {run.success ? (
                    <div className="px-4 py-3 space-y-3">
                      {/* Raw Prompt */}
                      <details>
                        <summary
                          className="text-[10px] font-mono uppercase tracking-wider cursor-pointer select-none hover:underline"
                          style={{ color: 'var(--text-subtle)' }}
                        >
                          Raw Prompt Sent
                        </summary>
                        <pre
                          className="mt-2 p-3 rounded text-xs font-mono whitespace-pre-wrap break-words"
                          style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }}
                        >
                          {baselineResult.prompt_used}
                        </pre>
                      </details>

                      {/* Model Response */}
                      <div>
                        <p
                          className="text-[10px] font-mono uppercase tracking-wider mb-2"
                          style={{ color: 'var(--text-subtle)' }}
                        >
                          Model Response
                        </p>
                        <div
                          className="p-3 rounded text-sm font-mono whitespace-pre-wrap break-words leading-relaxed"
                          style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }}
                        >
                          {run.response || '(empty)'}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="px-4 py-3">
                      <p className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                        {run.error || 'Run failed'}
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      {/* ═══ Technique Details Modal ═══ */}
      {expandedTechnique && expandedResult && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ background: 'rgba(0,0,0,0.25)' }}
          onClick={() => setExpandedTechnique(null)}
        >
          <div
            className="relative w-full max-w-2xl mx-4 rounded-lg shadow-xl overflow-hidden flex flex-col"
            style={{ background: 'var(--surface)', maxHeight: '85vh' }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div
              className="flex items-center justify-between px-6 py-4 shrink-0"
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

            <div className="p-6 space-y-6 overflow-y-auto">
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
                    {
                      label: expandedResult.scores?.consistency_is_provisional
                        ? 'CONSISTENCY*'
                        : 'CONSISTENCY',
                      value: expandedResult.scores?.consistency,
                    },
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
                        {formatScorePercent(s.value, 'PROV')}
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
                    {getDisplayPrompt(expandedResult.prompt)}
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
                    {expandedResult.response || 'No response captured for this technique run.'}
                    {expandedResult.error ? `\n\nError: ${expandedResult.error}` : ''}
                  </pre>
                </div>
              </div>

              {/* Individual Run Results */}
              {expandedResult.run_history && expandedResult.run_history.length > 1 && (
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <p
                      className="text-[11px] font-mono uppercase tracking-wider"
                      style={{ color: 'var(--text-subtle)' }}
                    >
                      Individual Run Results ({expandedResult.run_history.length} - Runs 1-{expandedResult.run_history.length})
                    </p>
                    <button
                      onClick={() => setShowIndividualRuns(!showIndividualRuns)}
                      className="px-3 py-1 rounded text-xs font-medium transition-colors"
                      style={{
                        border: '1px solid var(--blue)',
                        color: 'var(--blue)',
                        background: 'var(--surface)',
                      }}
                    >
                      {showIndividualRuns ? 'Hide' : 'Show'}
                    </button>
                  </div>
                  {showIndividualRuns && (
                    <div className="space-y-4">
                      {expandedResult.run_history.slice(0).map((run, idx) => (
                        <div
                        key={idx}
                        className="rounded-md p-4"
                        style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
                      >
                        {/* Run Header */}
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs font-semibold">
                              Run {run.run_index}
                            </span>
                            {run.success ? (
                              <span
                                className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                                style={{ background: '#f0fdf4', color: '#15803d', border: '1px solid #bbf7d0' }}
                              >
                                success
                              </span>
                            ) : (
                              <span
                                className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                                style={{ background: '#fef2f2', color: '#b91c1c', border: '1px solid #fecaca' }}
                              >
                                failed
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Metrics Grid */}
                        <div className="grid grid-cols-4 gap-2 mb-4">
                          <div className="p-2 rounded" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                            <p className="text-[9px] font-mono uppercase" style={{ color: 'var(--text-subtle)' }}>Overall</p>
                            <p className="text-lg font-mono font-light mt-1" style={{ color: scoreColor(run.scores?.overall ?? 0) }}>
                              {formatScorePercent(run.scores?.overall)}
                            </p>
                          </div>
                          <div className="p-2 rounded" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                            <p className="text-[9px] font-mono uppercase" style={{ color: 'var(--text-subtle)' }}>Accuracy</p>
                            <p className="text-lg font-mono font-light mt-1" style={{ color: scoreColor(run.scores?.accuracy ?? 0) }}>
                              {formatScorePercent(run.scores?.accuracy)}
                            </p>
                          </div>
                          <div className="p-2 rounded" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                            <p className="text-[9px] font-mono uppercase" style={{ color: 'var(--text-subtle)' }}>Consistency</p>
                            <p className="text-lg font-mono font-light mt-1" style={{ color: scoreColor(run.scores?.consistency ?? 0) }}>
                              {run.scores?.consistency === null || run.scores?.consistency === undefined
                                ? 'PROV'
                                : formatScorePercent(run.scores.consistency, 'PROV')}
                            </p>
                          </div>
                          <div className="p-2 rounded" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                            <p className="text-[9px] font-mono uppercase" style={{ color: 'var(--text-subtle)' }}>Efficiency</p>
                            <p className="text-lg font-mono font-light mt-1" style={{ color: scoreColor(run.scores?.efficiency ?? 0) }}>
                              {formatScorePercent(run.scores?.efficiency)}
                            </p>
                          </div>
                        </div>

                        {/* Raw Metrics */}
                        <div className="grid grid-cols-3 gap-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                          <div>
                            <span style={{ color: 'var(--text-subtle)' }}>Elapsed:</span> {run.metrics?.elapsed_time?.toFixed(2) ?? 'N/A'}s
                          </div>
                          <div>
                            <span style={{ color: 'var(--text-subtle)' }}>Tokens:</span> {run.metrics?.total_tokens ?? 'N/A'}
                          </div>
                          <div>
                            <span style={{ color: 'var(--text-subtle)' }}>Response:</span> {run.metrics?.completion_tokens ?? 'N/A'}
                          </div>
                        </div>

                        {/* Technique */}
                        <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
                          <p className="text-[9px] font-mono uppercase mb-1" style={{ color: 'var(--text-subtle)' }}>Technique</p>
                          <p className="text-sm font-mono font-semibold">{run.technique?.toUpperCase() ?? 'N/A'}</p>
                        </div>

                        {/* Prompt */}
                        <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
                          <p className="text-[9px] font-mono uppercase mb-1" style={{ color: 'var(--text-subtle)' }}>Prompt</p>
                          <pre className="text-xs font-mono whitespace-pre-wrap leading-relaxed max-h-32 overflow-y-auto p-2 rounded" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
                            {getDisplayPrompt(run.prompt)}
                          </pre>
                        </div>

                        {/* Response Preview */}
                        <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
                          <p className="text-[9px] font-mono uppercase mb-1" style={{ color: 'var(--text-subtle)' }}>Response</p>
                          <pre className="text-xs font-mono whitespace-pre-wrap leading-relaxed max-h-40 overflow-y-auto p-2 rounded" style={{ background: '#f0fdf4', border: '1px solid #bbf7d0' }}>
                            {run.response || 'No response'}
                          </pre>
                        </div>
                      </div>
                    ))}
                  </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      {/* Few-Shot Concepts Modal */}
      {showExampleTypes && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ background: 'rgba(0,0,0,0.4)' }}
          onClick={() => setShowExampleTypes(false)}
        >
          <div
            className="relative w-full max-w-2xl mx-4 rounded-xl shadow-2xl overflow-hidden"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="flex items-center justify-between px-5 py-4" style={{ borderBottom: '1px solid var(--border)' }}>
              <h3 className="text-sm font-semibold" style={{ color: 'var(--text)' }}>Available Few-Shot Concepts</h3>
              <button
                type="button"
                onClick={() => setShowExampleTypes(false)}
                className="text-lg leading-none px-1 rounded hover:opacity-70 transition-opacity"
                style={{ color: 'var(--text-muted)' }}
              >
                ✕
              </button>
            </div>
            {/* Subject tabs */}
            <div className="flex gap-1 px-5 pt-3 pb-2" style={{ borderBottom: '1px solid var(--border)' }}>
              {Object.keys(exampleTypes).map((subj) => {
                const label = subj.replace(/-/g, ' & ').replace(/\b\w/g, c => c.toUpperCase())
                const isActive = (activeConceptTab || Object.keys(exampleTypes)[0]) === subj
                return (
                  <button
                    key={subj}
                    type="button"
                    onClick={() => setActiveConceptTab(subj)}
                    className="px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
                    style={{
                      background: isActive ? 'var(--accent)' : 'var(--bg)',
                      color: isActive ? '#fff' : 'var(--text-muted)',
                      border: isActive ? 'none' : '1px solid var(--border)',
                    }}
                  >
                    {label}
                  </button>
                )
              })}
            </div>
            {/* Tab content */}
            <div className="px-5 py-4 overflow-y-auto text-xs space-y-3" style={{ maxHeight: '55vh' }}>
              {(() => {
                const currentTab = activeConceptTab || Object.keys(exampleTypes)[0]
                const diffs = exampleTypes[currentTab]
                if (!diffs) return null
                return Object.entries(diffs).map(([diff, types]) => (
                  <div key={diff} className="mb-2">
                    <p className="font-medium capitalize mb-1" style={{ color: 'var(--text-subtle)' }}>{diff}</p>
                    <ul className="ml-3 space-y-0.5">
                      {types.map((t) => (
                        <li key={t.concept} style={{ color: 'var(--text-muted)' }}>
                          {formatTypeName(t.concept)} <span className="opacity-60">({t.count} example{t.count !== 1 ? 's' : ''})</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))
              })()}
            </div>
          </div>
        </div>
      )}

      {/* ═══ Add Example Modal ═══ */}
      {showAddExample && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ background: 'rgba(0,0,0,0.4)' }}
          onClick={() => setShowAddExample(false)}
        >
          <div
            className="relative w-full max-w-2xl mx-4 rounded-xl shadow-2xl overflow-hidden flex flex-col"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)', maxHeight: '85vh' }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 shrink-0" style={{ borderBottom: '1px solid var(--border)' }}>
              <h3 className="text-sm font-semibold" style={{ color: 'var(--text)' }}>Add Few-Shot Example</h3>
              <button
                type="button"
                onClick={() => setShowAddExample(false)}
                className="text-lg leading-none px-1 rounded hover:opacity-70 transition-opacity"
                style={{ color: 'var(--text-muted)' }}
              >
                ✕
              </button>
            </div>

            {/* Body */}
            <div className="px-5 py-4 overflow-y-auto space-y-4">
              {/* ── Example 1 ── */}
              <div className="space-y-3">
                <p className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-subtle)' }}>Example 1</p>
                <div>
                  <label className="block text-xs font-medium mb-1" style={{ color: 'var(--text)' }}>Problem</label>
                  <textarea
                    value={addExProblem}
                    onChange={(e) => { setAddExProblem(e.target.value); setAddExAnalyzed(false) }}
                    rows={3}
                    placeholder="e.g., Solve for x: 3x - 7 = 20"
                    className="w-full px-3 py-2 rounded-md text-sm font-mono outline-none resize-y"
                    style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--bg)' }}
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium mb-1" style={{ color: 'var(--text)' }}>Solution</label>
                  <textarea
                    value={addExSolution}
                    onChange={(e) => setAddExSolution(e.target.value)}
                    rows={3}
                    placeholder="e.g., Add 7 to both sides: 3x = 27. Divide by 3: x = 9."
                    className="w-full px-3 py-2 rounded-md text-sm font-mono outline-none resize-y"
                    style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--bg)' }}
                  />
                </div>
              </div>

              {/* Divider */}
              <div style={{ borderTop: '1px solid var(--border)' }} />

              {/* ── Example 2 ── */}
              <div className="space-y-3">
                <p className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-subtle)' }}>Example 2 <span className="font-normal normal-case">(optional)</span></p>
                <div>
                  <label className="block text-xs font-medium mb-1" style={{ color: 'var(--text)' }}>Problem</label>
                  <textarea
                    value={addExProblem2}
                    onChange={(e) => { setAddExProblem2(e.target.value); setAddExAnalyzed2(false) }}
                    rows={3}
                    placeholder="e.g., Solve for y: 5y + 3 = 28"
                    className="w-full px-3 py-2 rounded-md text-sm font-mono outline-none resize-y"
                    style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--bg)' }}
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium mb-1" style={{ color: 'var(--text)' }}>Solution</label>
                  <textarea
                    value={addExSolution2}
                    onChange={(e) => setAddExSolution2(e.target.value)}
                    rows={3}
                    placeholder="e.g., Subtract 3 from both sides: 5y = 25. Divide by 5: y = 5."
                    className="w-full px-3 py-2 rounded-md text-sm font-mono outline-none resize-y"
                    style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--bg)' }}
                  />
                </div>
              </div>

              {/* Analyze button */}
              {!addExAnalyzed && (
                <button
                  type="button"
                  onClick={analyzeExample}
                  disabled={addExAnalyzing || !addExProblem.trim() || !addExSolution.trim()}
                  className="w-full py-2 rounded-md text-sm font-medium transition-colors"
                  style={{
                    background: (!addExProblem.trim() || !addExSolution.trim()) ? 'var(--border)' : 'var(--accent)',
                    color: (!addExProblem.trim() || !addExSolution.trim()) ? 'var(--text-muted)' : '#fff',
                    cursor: (!addExProblem.trim() || !addExSolution.trim()) ? 'not-allowed' : 'pointer',
                  }}
                >
                  {addExAnalyzing ? 'Asking Ollama...' : 'Auto-Detect Metadata'}
                </button>
              )}

              {/* Detected / editable metadata — Example 1 */}
              {addExAnalyzed && (
                <div className="space-y-3 p-3 rounded-md" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
                  <p className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-subtle)' }}>
                    Example 1 — Detected Metadata <span className="font-normal normal-case">(override if incorrect)</span>
                    {addExDetectionMethod && (
                      <span
                        className="ml-2 inline-block px-1.5 py-0.5 rounded text-[10px] font-medium normal-case"
                        style={{
                          background: addExDetectionMethod === 'ollama-llm' ? 'rgba(99,102,241,0.15)' : 'rgba(234,179,8,0.15)',
                          color: addExDetectionMethod === 'ollama-llm' ? 'rgb(99,102,241)' : 'rgb(161,128,17)',
                        }}
                      >
                        {addExDetectionMethod === 'ollama-llm' ? 'LLM' : 'Rule-based'}
                      </span>
                    )}
                  </p>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Subject</label>
                      <select
                        value={addExSubject}
                        onChange={(e) => setAddExSubject(e.target.value)}
                        className="w-full px-2 py-1.5 rounded text-sm outline-none"
                        style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--surface)' }}
                      >
                        <option value="algebra">Algebra</option>
                        <option value="counting-probability">Counting & Probability</option>
                        <option value="pre-calculus">Pre-Calculus</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Difficulty</label>
                      <select
                        value={addExDifficulty}
                        onChange={(e) => setAddExDifficulty(e.target.value)}
                        className="w-full px-2 py-1.5 rounded text-sm outline-none"
                        style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--surface)' }}
                      >
                        <option value="basic">Basic</option>
                        <option value="intermediate">Intermediate</option>
                        <option value="advanced">Advanced</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Type</label>
                      <input
                        value={addExType}
                        onChange={(e) => setAddExType(e.target.value)}
                        list="existing-types-list"
                        className="w-full px-2 py-1.5 rounded text-sm font-mono outline-none"
                        style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--surface)' }}
                        placeholder="e.g., solve_equation"
                      />
                      <datalist id="existing-types-list">
                        {(addExExistingTypes[addExSubject] || []).map((t) => (
                          <option key={t} value={t} />
                        ))}
                      </datalist>
                    </div>
                    <div>
                      <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Concept</label>
                      <input
                        value={addExConcept}
                        onChange={(e) => setAddExConcept(e.target.value)}
                        className="w-full px-2 py-1.5 rounded text-sm font-mono outline-none"
                        style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--surface)' }}
                        placeholder="e.g., linear_equation_solving"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Detected / editable metadata — Example 2 */}
              {addExAnalyzed2 && (
                <div className="space-y-3 p-3 rounded-md" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
                  <p className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-subtle)' }}>
                    Example 2 — Detected Metadata <span className="font-normal normal-case">(override if incorrect)</span>
                    {addExDetectionMethod2 && (
                      <span
                        className="ml-2 inline-block px-1.5 py-0.5 rounded text-[10px] font-medium normal-case"
                        style={{
                          background: addExDetectionMethod2 === 'ollama-llm' ? 'rgba(99,102,241,0.15)' : 'rgba(234,179,8,0.15)',
                          color: addExDetectionMethod2 === 'ollama-llm' ? 'rgb(99,102,241)' : 'rgb(161,128,17)',
                        }}
                      >
                        {addExDetectionMethod2 === 'ollama-llm' ? 'LLM' : 'Rule-based'}
                      </span>
                    )}
                  </p>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Subject</label>
                      <select
                        value={addExSubject2}
                        onChange={(e) => setAddExSubject2(e.target.value)}
                        className="w-full px-2 py-1.5 rounded text-sm outline-none"
                        style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--surface)' }}
                      >
                        <option value="algebra">Algebra</option>
                        <option value="counting-probability">Counting & Probability</option>
                        <option value="pre-calculus">Pre-Calculus</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Difficulty</label>
                      <select
                        value={addExDifficulty2}
                        onChange={(e) => setAddExDifficulty2(e.target.value)}
                        className="w-full px-2 py-1.5 rounded text-sm outline-none"
                        style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--surface)' }}
                      >
                        <option value="basic">Basic</option>
                        <option value="intermediate">Intermediate</option>
                        <option value="advanced">Advanced</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Type</label>
                      <input
                        value={addExType2}
                        onChange={(e) => setAddExType2(e.target.value)}
                        list="existing-types-list-2"
                        className="w-full px-2 py-1.5 rounded text-sm font-mono outline-none"
                        style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--surface)' }}
                        placeholder="e.g., solve_equation"
                      />
                      <datalist id="existing-types-list-2">
                        {(addExExistingTypes[addExSubject2] || []).map((t) => (
                          <option key={t} value={t} />
                        ))}
                      </datalist>
                    </div>
                    <div>
                      <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Concept</label>
                      <input
                        value={addExConcept2}
                        onChange={(e) => setAddExConcept2(e.target.value)}
                        className="w-full px-2 py-1.5 rounded text-sm font-mono outline-none"
                        style={{ border: '1px solid var(--border)', color: 'var(--text)', background: 'var(--surface)' }}
                        placeholder="e.g., linear_equation_solving"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Save button */}
              {addExAnalyzed && (
                <button
                  type="button"
                  onClick={saveExample}
                  disabled={addExSaving || !addExSolution.trim()}
                  className="w-full py-2 rounded-md text-sm font-medium transition-colors"
                  style={{
                    background: !addExSolution.trim() ? 'var(--border)' : '#16a34a',
                    color: !addExSolution.trim() ? 'var(--text-muted)' : '#fff',
                    cursor: !addExSolution.trim() ? 'not-allowed' : 'pointer',
                  }}
                >
                  {addExSaving ? 'Saving...' : `Save ${addExAnalyzed2 ? '2 Examples' : '1 Example'} to Bank`}
                </button>
              )}

              {/* Status message */}
              {addExStatus && (
                <div
                  className="px-3 py-2 rounded text-xs"
                  style={{
                    background: addExStatus.startsWith('Error') ? '#fef2f2' : '#f0fdf4',
                    border: `1px solid ${addExStatus.startsWith('Error') ? '#fecaca' : '#bbf7d0'}`,
                    color: addExStatus.startsWith('Error') ? '#dc2626' : '#16a34a',
                  }}
                >
                  {addExStatus}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
