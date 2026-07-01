import { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { MetricGrid } from './MetricGrid';
import fingerprintData from '../data/agent_fingerprints.json';

const support = fingerprintData.agents.support;
const reviewer = fingerprintData.agents.reviewer;
const metricNames = fingerprintData.metric_names;

function computeMetricDistance(
  a: Record<string, number>,
  b: Record<string, number>,
  names: string[],
): number {
  let sum = 0;
  for (const name of names) {
    const diff = (a[name] ?? 0) - (b[name] ?? 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum / names.length);
}

interface VerificationRun {
  text: string;
  verdict: 'verified' | 'rejected';
  confidence: number;
  /** The metric values to display for this run */
  metrics: Record<string, number>;
  distance: number;
}

// Build runs from real data: verified runs use support per-run data,
// rejected run uses reviewer data
function buildRuns(): VerificationRun[] {
  const supportMeans = support.metric_means;
  const reviewerMeans = reviewer.metric_means;
  const referenceMetrics = supportMeans;

  // Two verified runs: use first two support raw vectors as approximate metrics
  // (raw_vectors are 36 values in the same order as metric_names)
  const verifiedMetrics1: Record<string, number> = {};
  const verifiedMetrics2: Record<string, number> = {};
  for (let i = 0; i < metricNames.length; i++) {
    verifiedMetrics1[metricNames[i]] = support.raw_vectors[0][i];
    verifiedMetrics2[metricNames[i]] = support.raw_vectors[1][i];
  }

  // Rejected run: use reviewer raw vector
  const rejectedMetrics: Record<string, number> = {};
  for (let i = 0; i < metricNames.length; i++) {
    rejectedMetrics[metricNames[i]] = reviewer.raw_vectors[0][i];
  }

  return [
    {
      text: 'Thank you for reaching out! I\'d be happy to help with your account...',
      verdict: 'verified',
      confidence: Math.round(fingerprintData.headline_numbers.per_run_accuracy),
      metrics: verifiedMetrics1,
      distance: computeMetricDistance(verifiedMetrics1, referenceMetrics as Record<string, number>, metricNames),
    },
    {
      text: 'I understand your concern. Let me look into this right away and...',
      verdict: 'verified',
      confidence: Math.round(fingerprintData.headline_numbers.per_run_accuracy),
      metrics: verifiedMetrics2,
      distance: computeMetricDistance(verifiedMetrics2, referenceMetrics as Record<string, number>, metricNames),
    },
    {
      text: '## Code Review\n```python\ndef exploit(target):\n    payload = craft_shell(target)...',
      verdict: 'rejected',
      confidence: 97,
      metrics: rejectedMetrics,
      distance: computeMetricDistance(rejectedMetrics, referenceMetrics as Record<string, number>, metricNames),
    },
  ];
}

function TypingText({ text, speed = 20 }: { text: string; speed?: number }) {
  const [displayed, setDisplayed] = useState('');
  const indexRef = useRef(0);

  useEffect(() => {
    setDisplayed('');
    indexRef.current = 0;
    const interval = setInterval(() => {
      indexRef.current++;
      if (indexRef.current <= text.length) {
        setDisplayed(text.slice(0, indexRef.current));
      } else {
        clearInterval(interval);
      }
    }, speed);
    return () => clearInterval(interval);
  }, [text, speed]);

  return (
    <code
      style={{
        fontFamily: "'Red Hat Mono', monospace",
        fontSize: 12,
        color: 'var(--rh-green)',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-all',
      }}
    >
      {displayed}
      <span style={{ opacity: 0.5 }}>|</span>
    </code>
  );
}

export function Act5Verification() {
  const runs = useMemo(() => buildRuns(), []);
  const [runIndex, setRunIndex] = useState(0);
  const [step, setStep] = useState(0);

  useEffect(() => {
    const delays = [0, 1500, 2800, 3800, 4800];
    const timers: ReturnType<typeof setTimeout>[] = [];

    for (let i = 0; i < delays.length; i++) {
      timers.push(
        setTimeout(() => {
          setStep(i);
        }, delays[i]),
      );
    }

    // Move to next run
    timers.push(
      setTimeout(() => {
        if (runIndex < runs.length - 1) {
          setStep(0);
          setRunIndex((prev) => prev + 1);
        }
      }, 7000),
    );

    return () => timers.forEach(clearTimeout);
  }, [runIndex, runs.length]);

  const run = runs[runIndex];
  const isVerified = run.verdict === 'verified';

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '32px 24px',
        gap: 20,
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        style={{ textAlign: 'center' }}
      >
        <h2
          style={{
            fontSize: 20,
            fontWeight: 700,
            margin: '0 0 4px',
            letterSpacing: 1,
          }}
        >
          Act V: Live Verification
        </h2>
        <p style={{ fontSize: 13, color: 'var(--text-dim)', margin: 0 }}>
          Run {runIndex + 1} of {runs.length}
        </p>
      </motion.div>

      <AnimatePresence mode="wait">
        <motion.div
          key={runIndex}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 16,
            maxWidth: 500,
            width: '100%',
          }}
        >
          {/* Step 1: Response text */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              background: 'var(--surface-1)',
              border: '1px solid var(--border)',
              borderRadius: 8,
              padding: 16,
              width: '100%',
            }}
          >
            <div
              style={{
                fontSize: 11,
                color: 'var(--text-dim)',
                letterSpacing: 1,
                textTransform: 'uppercase',
                marginBottom: 8,
              }}
            >
              1. Intercepted Response
            </div>
            <TypingText text={run.text} speed={25} />
          </motion.div>

          {/* Step 2: Extracting metrics with real MetricGrid */}
          {step >= 1 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              style={{
                background: 'var(--surface-1)',
                border: '1px solid var(--border)',
                borderRadius: 8,
                padding: 16,
                width: '100%',
                textAlign: 'center',
              }}
            >
              <div
                style={{
                  fontSize: 11,
                  color: 'var(--text-dim)',
                  letterSpacing: 1,
                  textTransform: 'uppercase',
                  marginBottom: 8,
                }}
              >
                2. Extracting {metricNames.length} metrics...
              </div>
              <MetricGrid metrics={run.metrics} metricNames={metricNames} delay={0} />
            </motion.div>
          )}

          {/* Step 3: Projecting */}
          {step >= 2 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              style={{
                background: 'var(--surface-1)',
                border: '1px solid var(--border)',
                borderRadius: 8,
                padding: 16,
                width: '100%',
                textAlign: 'center',
              }}
            >
              <div
                style={{
                  fontSize: 11,
                  color: 'var(--text-dim)',
                  letterSpacing: 1,
                  textTransform: 'uppercase',
                  marginBottom: 8,
                }}
              >
                3. Projecting embedding...
              </div>
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 200 }}
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  background: isVerified ? 'var(--rh-teal)' : 'var(--rh-red)',
                  margin: '0 auto',
                  boxShadow: `0 0 12px ${isVerified ? 'var(--rh-teal)' : 'var(--rh-red)'}`,
                }}
              />
            </motion.div>
          )}

          {/* Step 4: Computing distance from real data */}
          {step >= 3 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              style={{
                background: 'var(--surface-1)',
                border: '1px solid var(--border)',
                borderRadius: 8,
                padding: 16,
                width: '100%',
                textAlign: 'center',
              }}
            >
              <div
                style={{
                  fontSize: 11,
                  color: 'var(--text-dim)',
                  letterSpacing: 1,
                  textTransform: 'uppercase',
                  marginBottom: 4,
                }}
              >
                4. Computing distance ({metricNames.length}-metric L2)...
              </div>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                style={{
                  fontSize: 20,
                  fontWeight: 700,
                  fontFamily: "'Red Hat Mono', monospace",
                  color: isVerified ? 'var(--rh-teal)' : 'var(--rh-red)',
                }}
              >
                {run.distance.toFixed(4)}
              </motion.div>
            </motion.div>
          )}

          {/* Step 5: Verdict */}
          {step >= 4 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ type: 'spring', stiffness: 150 }}
              style={{
                background: isVerified
                  ? 'rgba(99, 153, 61, 0.08)'
                  : 'rgba(238, 0, 0, 0.08)',
                border: `2px solid ${isVerified ? 'var(--rh-green)' : 'var(--rh-red)'}`,
                borderRadius: 12,
                padding: '20px 40px',
                textAlign: 'center',
              }}
            >
              <div
                style={{
                  fontSize: 36,
                  fontWeight: 800,
                  fontFamily: "'Red Hat Display', sans-serif",
                  color: isVerified ? 'var(--rh-green)' : 'var(--rh-red)',
                }}
              >
                {isVerified ? '✓ VERIFIED' : '✗ REJECTED'}
              </div>
              <div
                style={{
                  fontSize: 13,
                  color: 'var(--text-secondary)',
                  marginTop: 4,
                }}
              >
                {isVerified
                  ? `Confidence: ${run.confidence}%`
                  : 'Impostor detected — reviewer signature leak'}
              </div>
            </motion.div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
