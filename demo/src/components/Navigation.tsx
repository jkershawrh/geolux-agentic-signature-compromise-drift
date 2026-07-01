import { motion } from 'motion/react';

const ACT_TITLES = [
  'Fingerprint',
  'Comparison',
  'Impostor',
  'Embeddings',
  'Verification',
  'Results',
];

interface NavigationProps {
  total: number;
  current: number;
  onNext: () => void;
  onBack: () => void;
}

export function Navigation({ total, current, onNext, onBack }: NavigationProps) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 24px',
        background: 'var(--surface-1)',
        borderTop: '1px solid var(--border)',
        minHeight: 56,
      }}
    >
      <motion.button
        onClick={onBack}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        style={{
          background: 'none',
          border: '1px solid var(--border)',
          color: current > 0 ? 'var(--text-primary)' : 'var(--text-disabled)',
          cursor: current > 0 ? 'pointer' : 'default',
          padding: '8px 16px',
          borderRadius: 6,
          fontSize: 13,
          fontFamily: "'Red Hat Text', sans-serif",
          width: 100,
        }}
        disabled={current === 0}
      >
        &larr; Back
      </motion.button>

      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {Array.from({ length: total }).map((_, i) => (
            <div
              key={i}
              style={{
                width: i === current ? 10 : 6,
                height: i === current ? 10 : 6,
                borderRadius: '50%',
                background: i === current ? 'var(--text-primary)' : 'var(--text-disabled)',
                transition: 'all 0.3s ease',
              }}
            />
          ))}
        </div>
        <span
          style={{
            fontSize: 12,
            color: 'var(--text-dim)',
            letterSpacing: 1,
            textTransform: 'uppercase',
            minWidth: 100,
            textAlign: 'center',
          }}
        >
          {ACT_TITLES[current]}
        </span>
      </div>

      <motion.button
        onClick={onNext}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        style={{
          background: current < total - 1 ? 'var(--rh-red)' : 'var(--text-disabled)',
          border: 'none',
          color: 'white',
          cursor: current < total - 1 ? 'pointer' : 'default',
          padding: '8px 16px',
          borderRadius: 6,
          fontSize: 13,
          fontFamily: "'Red Hat Text', sans-serif",
          fontWeight: 600,
          width: 120,
        }}
        disabled={current >= total - 1}
      >
        Continue &rarr;
      </motion.button>
    </div>
  );
}
