import { motion } from 'motion/react';

const stats = [
  { value: '3.6%', label: 'EER', color: 'var(--rh-red)' },
  { value: '0.992', label: 'ROC AUC', color: 'var(--rh-teal)' },
  { value: '93%', label: 'Per-Run', color: 'var(--rh-green)' },
  { value: '100%', label: 'Batch', color: 'var(--rh-blue)' },
  { value: '7', label: 'Models', color: 'var(--rh-purple)' },
  { value: '19', label: 'Agents', color: 'var(--rh-orange)' },
  { value: '382', label: 'Tests', color: 'var(--rh-teal)' },
  { value: '36', label: 'Metrics', color: 'var(--rh-yellow)' },
];

export function Act6Results() {
  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '32px 24px',
        gap: 32,
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
          Act VI: The Verdict
        </h2>
        <p style={{ fontSize: 13, color: 'var(--text-dim)', margin: 0 }}>
          Evaluation results across all agents and models
        </p>
      </motion.div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: 16,
          maxWidth: 600,
          width: '100%',
        }}
      >
        {stats.map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ delay: 0.2 + i * 0.1, duration: 0.5 }}
            style={{
              background: 'var(--surface-1)',
              border: '1px solid var(--border)',
              borderRadius: 8,
              padding: '20px 12px',
              textAlign: 'center',
            }}
          >
            <div
              style={{
                fontSize: 28,
                fontWeight: 800,
                fontFamily: "'Red Hat Display', sans-serif",
                color: stat.color,
                lineHeight: 1,
              }}
            >
              {stat.value}
            </div>
            <div
              style={{
                fontSize: 11,
                color: 'var(--text-dim)',
                letterSpacing: 1,
                textTransform: 'uppercase',
                marginTop: 8,
              }}
            >
              {stat.label}
            </div>
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 0.8 }}
        style={{
          maxWidth: 520,
          textAlign: 'center',
        }}
      >
        <p
          style={{
            fontSize: 18,
            fontWeight: 600,
            fontFamily: "'Red Hat Display', sans-serif",
            color: 'var(--text-primary)',
            lineHeight: 1.6,
            margin: '0 0 24px',
          }}
        >
          Geometric fingerprints make agent identity verifiable, tamper-evident,
          and drift-aware &mdash; without modifying the agent itself.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2, duration: 0.8 }}
        style={{
          fontSize: 12,
          color: 'var(--text-disabled)',
          letterSpacing: 2,
          textTransform: 'uppercase',
          textAlign: 'center',
        }}
      >
        GEOLUX &mdash; Agent Signature, Compromise &amp; Drift
      </motion.div>
    </div>
  );
}
