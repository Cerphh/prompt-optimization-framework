'use client'

import { useState } from 'react'

interface HelpGuideProps {
  open: boolean
  onClose: () => void
}

interface Step {
  title: string
  body: React.ReactNode
}

const STEPS: Step[] = [
  {
    title: 'Welcome! What is this app?',
    body: (
      <>
        <p className="mb-3">
          This is a <strong>prompt testing playground</strong> for math problems.
        </p>
        <p className="mb-3">
          You give it a math question. It asks an AI model the same question in
          several different ways (these are called <em>prompting techniques</em>).
          It then shows you which way of asking gave the best, fastest, and most
          consistent answer.
        </p>
        <p className="mb-3">
          Think of it like trying out different ways to phrase a question to a
          tutor — and seeing which phrasing helps them solve it best.
        </p>
        <div
          className="mt-4 p-3 rounded-md text-sm"
          style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
        >
          <strong>In plain words:</strong> type a math problem &rarr; press Run
          &rarr; see which way of asking the AI worked best.
        </div>
      </>
    ),
  },
  {
    title: 'Step 1 — Pick a Run Mode',
    body: (
      <>
        <p className="mb-3">
          The <strong>Run Mode</strong> dropdown lets you choose how thorough
          the test should be:
        </p>
        <ul className="space-y-3 text-sm">
          <li>
            <strong>Normal mode</strong> &mdash; Quick test. The app picks the
            best technique for you based on past results, so you get an answer
            fast.
            <span className="block mt-1" style={{ color: 'var(--text-muted)' }}>
              Use this when you just want a good answer.
            </span>
          </li>
          <li>
            <strong>Benchmark mode</strong> &mdash; Full comparison. Tries
            <em> every</em> technique and ranks them. You must provide the
            correct answer so the app can grade them fairly.
            <span className="block mt-1" style={{ color: 'var(--text-muted)' }}>
              Use this when you want to compare techniques scientifically.
            </span>
          </li>
          <li>
            <strong>Baseline mode</strong> &mdash; A plain &ldquo;no tricks&rdquo;
            run. Just asks the AI directly with no special prompting. Useful as
            a reference point.
            <span className="block mt-1" style={{ color: 'var(--text-muted)' }}>
              Use this to see what the model does without any optimization.
            </span>
          </li>
        </ul>
      </>
    ),
  },
  {
    title: 'Step 2 — Subject and Difficulty',
    body: (
      <>
        <p className="mb-3">
          <strong>Subject Category</strong> tells the app what kind of math
          you&rsquo;re asking about (Algebra, Probability, Pre-calculus). The
          app tries to detect this automatically from your question, but you
          can change it.
        </p>
        <p className="mb-3">
          <strong>Difficulty</strong> is a hint about how hard the problem is:
        </p>
        <ul className="text-sm space-y-1 ml-5 list-disc">
          <li><strong>Basic</strong> &mdash; one or two simple steps</li>
          <li><strong>Intermediate</strong> &mdash; several steps, some setup</li>
          <li><strong>Advanced</strong> &mdash; multi-step, tricky reasoning</li>
        </ul>
        <p className="mt-3 text-sm" style={{ color: 'var(--text-muted)' }}>
          You don&rsquo;t have to be precise &mdash; the app uses these only as
          hints to help pick the right strategy.
        </p>
      </>
    ),
  },
  {
    title: 'Step 3 — Type your problem',
    body: (
      <>
        <p className="mb-3">
          Just type your math problem in plain English or with math symbols.
          Examples that work well:
        </p>
        <ul
          className="text-sm space-y-2 p-3 rounded-md"
          style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
        >
          <li className="font-mono">Solve for x: 2x + 5 = 15</li>
          <li className="font-mono">What is the probability of rolling a 6 twice in a row?</li>
          <li className="font-mono">Find the derivative of f(x) = 3x² + 2x</li>
        </ul>
        <p className="mt-3 text-sm">
          Need a special symbol like <span className="font-mono">π</span>,{' '}
          <span className="font-mono">√</span>, or{' '}
          <span className="font-mono">∑</span>? Click the{' '}
          <span className="font-mono px-1.5 py-0.5 rounded" style={{ background: 'var(--border)' }}>Ω</span>{' '}
          button at the top right of the text box.
        </p>
        <p className="mt-3 text-sm" style={{ color: 'var(--text-muted)' }}>
          For Benchmark or Baseline mode, also fill in the{' '}
          <strong>Expected Answer</strong> box so the app can grade the AI&rsquo;s
          response.
        </p>
      </>
    ),
  },
  {
    title: 'Step 4 — Read the results',
    body: (
      <>
        <p className="mb-3">After running, you&rsquo;ll see a results table with these scores:</p>
        <ul className="space-y-2 text-sm">
          <li>
            <strong>Accuracy</strong> &mdash; Did the AI get the right answer?
            (Higher is better.)
          </li>
          <li>
            <strong>Consistency</strong> &mdash; If we ask multiple times, does
            it keep giving the same answer? (Higher = more reliable.)
          </li>
          <li>
            <strong>Efficiency</strong> &mdash; How fast and concise was the
            answer? (Higher = quicker, less wordy.)
          </li>
          <li>
            <strong>Overall</strong> &mdash; A weighted combination of the
            above. The technique with the highest Overall score wins.
          </li>
        </ul>
        <div
          className="mt-4 p-3 rounded-md text-sm"
          style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
        >
          <strong>Color hints:</strong>{' '}
          <span style={{ color: 'var(--green)' }}>green = excellent</span>,{' '}
          <span style={{ color: 'var(--blue)' }}>blue = good</span>,{' '}
          <span style={{ color: 'var(--amber)' }}>amber = okay</span>, gray =
          weak.
        </div>
      </>
    ),
  },
  {
    title: 'What are &ldquo;prompting techniques&rdquo;?',
    body: (
      <>
        <p className="mb-3">
          A <strong>prompting technique</strong> is a way of phrasing a question
          to an AI to help it think better. Examples:
        </p>
        <ul className="space-y-2 text-sm">
          <li>
            <strong>Zero-shot</strong> &mdash; Just ask the question directly,
            no examples.
          </li>
          <li>
            <strong>Few-shot</strong> &mdash; Show the AI a couple of solved
            examples first, then ask your question.
          </li>
          <li>
            <strong>Chain-of-Thought (CoT)</strong> &mdash; Ask the AI to
            &ldquo;think step by step&rdquo; before answering.
          </li>
          <li>
            <strong>Role-Based</strong> &mdash; Give the AI a role (for example,
            &ldquo;act like a careful math tutor&rdquo;) before asking the problem.
          </li>
        </ul>
        <p className="mt-3 text-sm" style={{ color: 'var(--text-muted)' }}>
          Different techniques work better for different problem types. This
          app helps you discover which one fits your problem best.
        </p>
      </>
    ),
  },
  {
    title: 'You&rsquo;re ready!',
    body: (
      <>
        <p className="mb-3">That&rsquo;s the whole tour. To recap:</p>
        <ol className="space-y-2 text-sm ml-5 list-decimal">
          <li>Pick a <strong>Run Mode</strong> (start with Normal).</li>
          <li>Type your math problem.</li>
          <li>Press <strong>Run</strong>.</li>
          <li>Compare the scores in the results table.</li>
        </ol>
        <p className="mt-4 text-sm" style={{ color: 'var(--text-muted)' }}>
          You can re-open this guide anytime by clicking the{' '}
          <strong>Help</strong> button in the top-right corner.
        </p>
      </>
    ),
  },
]

export default function HelpGuide({ open, onClose }: HelpGuideProps) {
  const [step, setStep] = useState(0)

  if (!open) return null

  const current = STEPS[step]
  const isFirst = step === 0
  const isLast = step === STEPS.length - 1

  const handleClose = () => {
    setStep(0)
    onClose()
  }

  return (
    <div
      onClick={handleClose}
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.45)' }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-[560px] rounded-xl shadow-xl flex flex-col"
        style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          maxHeight: '85vh',
        }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-5 py-3 shrink-0"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          <div className="flex items-center gap-2">
            <span
              className="inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-semibold"
              style={{ background: 'var(--accent)', color: '#fff' }}
            >
              ?
            </span>
            <span className="text-sm font-semibold">Quick Guide</span>
            <span
              className="text-xs font-mono ml-2"
              style={{ color: 'var(--text-muted)' }}
            >
              {step + 1} / {STEPS.length}
            </span>
          </div>
          <button
            type="button"
            onClick={handleClose}
            className="text-sm px-2 py-1 rounded"
            style={{ color: 'var(--text-muted)', background: 'transparent', border: 'none', cursor: 'pointer' }}
            aria-label="Close guide"
          >
            ✕
          </button>
        </div>

        {/* Progress dots */}
        <div className="flex items-center gap-1.5 px-5 pt-3 shrink-0">
          {STEPS.map((_, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setStep(i)}
              aria-label={`Go to step ${i + 1}`}
              className="h-1.5 rounded-full transition-all"
              style={{
                width: i === step ? '20px' : '8px',
                background: i === step ? 'var(--accent)' : 'var(--border-strong, var(--border))',
                border: 'none',
                cursor: 'pointer',
              }}
            />
          ))}
        </div>

        {/* Body */}
        <div className="px-5 py-4 overflow-y-auto flex-1">
          <h3 className="text-lg font-semibold mb-3" dangerouslySetInnerHTML={{ __html: current.title }} />
          <div className="text-sm leading-relaxed" style={{ color: 'var(--text)' }}>
            {current.body}
          </div>
        </div>

        {/* Footer */}
        <div
          className="flex items-center justify-between px-5 py-3 shrink-0"
          style={{ borderTop: '1px solid var(--border)' }}
        >
          <button
            type="button"
            onClick={handleClose}
            className="text-xs"
            style={{ color: 'var(--text-muted)', background: 'transparent', border: 'none', cursor: 'pointer' }}
          >
            Skip tour
          </button>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setStep((s) => Math.max(0, s - 1))}
              disabled={isFirst}
              className="px-3 py-1.5 rounded-md text-sm font-medium"
              style={{
                background: 'transparent',
                color: isFirst ? 'var(--text-subtle)' : 'var(--text)',
                border: '1px solid var(--border)',
                cursor: isFirst ? 'not-allowed' : 'pointer',
              }}
            >
              ← Back
            </button>
            {isLast ? (
              <button
                type="button"
                onClick={handleClose}
                className="px-4 py-1.5 rounded-md text-sm font-medium"
                style={{ background: 'var(--accent)', color: '#fff', border: 'none', cursor: 'pointer' }}
              >
                Got it!
              </button>
            ) : (
              <button
                type="button"
                onClick={() => setStep((s) => Math.min(STEPS.length - 1, s + 1))}
                className="px-4 py-1.5 rounded-md text-sm font-medium"
                style={{ background: 'var(--accent)', color: '#fff', border: 'none', cursor: 'pointer' }}
              >
                Next →
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
