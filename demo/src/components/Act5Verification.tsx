import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';

interface VerificationRun {
  text: string;
  verdict: 'verified' | 'rejected';
  confidence: number;
}

const RUNS: VerificationRun[] = [
  {
    text: 'The capital of France is Paris. Is there anything else?',
    verdict: 'verified',
    confidence: 94,
  },
  {
    text: 'Gravity is a fundamental force that attracts objects with mass...',
    verdict: 'verified',
    confidence: 91,
  },
  {
    text: '## Finding 1\n```python\ndef exploit(target):\n    payload = craft_shell(target)...',
    verdict: 'rejected',
    confidence: 97,
  },
];

function MetricBars({ visible }: { visible: boolean }) {
  return (
    <div
      style={{
        display: 'flex',
        gap: 2,
        flexWrap: 'wrap',
        maxWidth: 360,
        justifyContent: 'center',
      }}
    >
      {Array.from({ length: 36 }).map((_, i) => (
        <motion.div
          key={i}
          initial={{ scaleY: 0 }}
          animate={visible ? { scaleY: 1 } : { scaleY: 0 }}
          transition={{ delay: i * 0.02, duration: 0.3 }}
          style={{
            width: 8,
            height: 16 + Math.random() * 16,
            background: `hsl(${170 + i * 3}, 60%, 50%)`,
            borderRadius: 2,
            transformOrigin: 'bottom',
            opacity: 0.7,
          }}
        />
      ))}
    </div>
  );
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
  const [runIndex, setRunIndex] = useState(0);
  const [step, setStep] = useState(0);

  useEffect(() => {
    const steps = [0, 1, 2, 3, 4];
    const delays = [0, 1500, 2800, 3800, 4800];
    const timers: ReturnType<typeof setTimeout>[] = [];

    for (let i = 0; i < steps.length; i++) {
      timers.push(
        setTimeout(() => {
          setStep(steps[i]);
        }, delays[i]),
      );
    }

    // Move to next run
    timers.push(
      setTimeout(() => {
        if (runIndex < RUNS.length - 1) {
          setStep(0);
          setRunIndex((prev) => prev + 1);
        }
      }, 7000),
    );

    return () => timers.forEach(clearTimeout);
  }, [runIndex]);

  const run = RUNS[runIndex];
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
          Run {runIndex + 1} of {RUNS.length}
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

          {/* Step 2: Extracting metrics */}
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
                2. Extracting 36 metrics...
              </div>
              <MetricBars visible={step >= 1} />
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

          {/* Step 4: Computing distance */}
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
                4. Computing distance...
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
                {isVerified ? '0.042' : '0.873'}
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
                  : 'Impostor detected'}
              </div>
            </motion.div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
