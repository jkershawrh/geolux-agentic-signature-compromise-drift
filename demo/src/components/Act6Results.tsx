import { motion } from 'motion/react';
import fingerprintData from '../data/agent_fingerprints.json';

const h = fingerprintData.headline_numbers;

const stats = [
  { value: `${h.eer}%`, label: 'EER', color: 'var(--rh-red)' },
  { value: h.auc.toFixed(3), label: 'ROC AUC', color: 'var(--rh-teal)' },
  { value: `${h.per_run_accuracy}%`, label: 'Per-Run', color: 'var(--rh-green)' },
  { value: `${h.batch_accuracy}%`, label: 'Batch', color: 'var(--rh-blue)' },
  { value: String(h.models_validated), label: 'Models', color: 'var(--rh-purple)' },
  { value: String(h.agents_tested), label: 'Agents', color: 'var(--rh-orange)' },
  { value: String(h.tests), label: 'Tests', color: 'var(--rh-teal)' },
  { value: String(h.metrics), label: 'Metrics', color: 'var(--rh-yellow)' },
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

      {/* Additional detail stats */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.2, duration: 0.5 }}
        style={{
          display: 'flex',
          gap: 16,
          justifyContent: 'center',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ textAlign: 'center', padding: '8px 16px' }}>
          <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'Red Hat Display', sans-serif", color: 'var(--rh-teal)' }}>
            {h.fisher_ratio.toFixed(2)}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: 1, textTransform: 'uppercase' }}>Fisher Ratio</div>
        </div>
        <div style={{ textAlign: 'center', padding: '8px 16px' }}>
          <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'Red Hat Display', sans-serif", color: 'var(--rh-orange)' }}>
            {h.cohens_d.toFixed(2)}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: 1, textTransform: 'uppercase' }}>Cohen&apos;s d</div>
        </div>
        <div style={{ textAlign: 'center', padding: '8px 16px' }}>
          <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'Red Hat Display', sans-serif", color: 'var(--rh-red)' }}>
            {'±'}{h.eer_ci}%
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: 1, textTransform: 'uppercase' }}>EER 95% CI</div>
        </div>
      </motion.div>

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
