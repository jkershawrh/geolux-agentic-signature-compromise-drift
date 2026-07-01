import { motion } from 'motion/react';
import { RadarChart } from './RadarChart';

const ALPHA_VALUES = [0.3, 0.4, 0.0, 0.2, 0.8, 0.8, 0.0, 0.9, 0.7];
const BETA_VALUES = [0.8, 0.6, 0.7, 0.6, 0.9, 0.7, 0.1, 0.5, 0.8];

const stats = [
  { label: 'Fisher Ratio', value: '4.23', color: 'var(--rh-teal)' },
  { label: "Cohen's d", value: '2.17', color: 'var(--rh-orange)' },
  { label: 'p-value', value: '< 0.001', color: 'var(--rh-green)' },
];

export function Act2TwoAgents() {
  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '32px 24px',
        gap: 24,
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
          Act II: Two Agents
        </h2>
        <p style={{ fontSize: 13, color: 'var(--text-dim)', margin: 0 }}>
          Side-by-side fingerprint comparison
        </p>
      </motion.div>

      <div
        style={{
          display: 'flex',
          gap: 32,
          alignItems: 'center',
          flexWrap: 'wrap',
          justifyContent: 'center',
        }}
      >
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          style={{ textAlign: 'center' }}
        >
          <p
            style={{
              fontSize: 12,
              color: 'var(--rh-teal)',
              letterSpacing: 2,
              margin: '0 0 8px',
              textTransform: 'uppercase',
            }}
          >
            Alpha &mdash; Support
          </p>
          <RadarChart values={ALPHA_VALUES} color="var(--rh-teal)" animate={true} size={260} />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          style={{ textAlign: 'center' }}
        >
          <p
            style={{
              fontSize: 12,
              color: 'var(--rh-orange)',
              letterSpacing: 2,
              margin: '0 0 8px',
              textTransform: 'uppercase',
            }}
          >
            Beta &mdash; Reviewer
          </p>
          <RadarChart values={BETA_VALUES} color="var(--rh-orange)" animate={true} size={260} />
        </motion.div>
      </div>

      <div
        style={{
          display: 'flex',
          gap: 16,
          flexWrap: 'wrap',
          justifyContent: 'center',
        }}
      >
        {stats.map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1 + i * 0.2, duration: 0.5 }}
            style={{
              background: 'var(--surface-1)',
              border: '1px solid var(--border)',
              borderRadius: 8,
              padding: '16px 24px',
              textAlign: 'center',
              minWidth: 120,
            }}
          >
            <div
              style={{
                fontSize: 24,
                fontWeight: 700,
                fontFamily: "'Red Hat Display', sans-serif",
                color: stat.color,
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
                marginTop: 4,
              }}
            >
              {stat.label}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
