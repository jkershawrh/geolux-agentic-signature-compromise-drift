import { motion } from 'motion/react';

export function HeroScreen({ onStart }: { onStart: () => void }) {
  return (
    <div
      onClick={onStart}
      style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
      }}
    >
      <motion.div
        style={{
          fontSize: 14,
          letterSpacing: 4,
          color: 'var(--text-dim)',
          marginBottom: 24,
          textTransform: 'uppercase',
        }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        RED HAT &times; IBM &times; INTEL
      </motion.div>
      <motion.div
        style={{
          fontSize: 28,
          fontWeight: 800,
          fontFamily: "'Red Hat Display', sans-serif",
          letterSpacing: 3,
          marginBottom: 12,
        }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.8 }}
      >
        GEOLUX
      </motion.div>
      <motion.div
        style={{
          fontSize: 14,
          color: 'var(--text-secondary)',
          letterSpacing: 1,
          marginBottom: 8,
        }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 0.8 }}
      >
        Agent Signature Verification
      </motion.div>
      <motion.div
        style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 48 }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.3, duration: 0.8 }}
      >
        Can you prove an AI agent is who it claims to be?
      </motion.div>
      <motion.div
        style={{ fontSize: 11, color: 'var(--text-disabled)', letterSpacing: 2 }}
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 1, 0] }}
        transition={{ delay: 2, duration: 2, repeat: Infinity }}
      >
        CLICK TO BEGIN
      </motion.div>
    </div>
  );
}
