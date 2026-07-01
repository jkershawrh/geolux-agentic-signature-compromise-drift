import { motion } from 'motion/react';
import { RadarChart } from './RadarChart';

const ALPHA_VALUES = [0.3, 0.4, 0.0, 0.2, 0.8, 0.8, 0.0, 0.9, 0.7];

export function Act1Fingerprint() {
  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px 24px',
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
          Act I: The Fingerprint
        </h2>
        <p style={{ fontSize: 13, color: 'var(--text-dim)', margin: 0 }}>
          Agent Alpha &mdash; Customer Support
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3, duration: 0.6 }}
      >
        <RadarChart values={ALPHA_VALUES} color="var(--rh-teal)" animate={true} size={320} />
      </motion.div>

      <div
        style={{
          maxWidth: 500,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}
      >
        {[
          'Every agent develops a unique behavioral fingerprint across 9 dimensions.',
          'These dimensions capture how an agent structures responses, uses tokens, and reasons through problems.',
          '36 individual metrics are extracted per interaction, then projected into a dense embedding.',
        ].map((text, i) => (
          <motion.p
            key={i}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 1 + i * 0.3, duration: 0.5 }}
            style={{
              fontSize: 13,
              color: 'var(--text-secondary)',
              margin: 0,
              paddingLeft: 16,
              borderLeft: '2px solid var(--rh-teal)',
            }}
          >
            {text}
          </motion.p>
        ))}
      </div>
    </div>
  );
}
