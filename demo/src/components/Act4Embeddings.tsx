import { useMemo } from 'react';
import { motion } from 'motion/react';
import { ScatterPlot } from './ScatterPlot';

const clusters = [
  { name: 'Support', color: '#37a3a3', cx: 150, cy: 200, r: 30 },
  { name: 'Reviewer', color: '#f0561d', cx: 350, cy: 100, r: 35 },
  { name: 'Analyst', color: '#5e40be', cx: 300, cy: 350, r: 25 },
  { name: 'Auditor', color: '#63993d', cx: 450, cy: 280, r: 30 },
  { name: 'Clinical', color: '#0066cc', cx: 180, cy: 380, r: 28 },
];

function generateClusterPoints() {
  const points: { x: number; y: number; color: string; cluster: string }[] = [];
  for (const cluster of clusters) {
    const count = 12 + Math.floor(Math.random() * 8);
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const dist = Math.random() * cluster.r;
      points.push({
        x: cluster.cx + Math.cos(angle) * dist,
        y: cluster.cy + Math.sin(angle) * dist,
        color: cluster.color,
        cluster: cluster.name,
      });
    }
  }
  return points;
}

export function Act4Embeddings() {
  const points = useMemo(() => generateClusterPoints(), []);
  const centroids = useMemo(
    () =>
      clusters.map((c) => ({
        x: c.cx,
        y: c.cy,
        color: c.color,
        label: c.name,
      })),
    [],
  );

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
          Act IV: The Embedding Space
        </h2>
        <p style={{ fontSize: 13, color: 'var(--text-dim)', margin: 0 }}>
          Agent signatures projected into 2D space
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3, duration: 0.6 }}
        style={{
          background: 'var(--surface-1)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: 16,
        }}
      >
        <ScatterPlot
          points={points}
          centroids={centroids}
          animate={true}
          width={580}
          height={440}
          showAUC={true}
        />
      </motion.div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2.5, duration: 0.8 }}
        style={{
          fontSize: 13,
          color: 'var(--text-secondary)',
          margin: 0,
          maxWidth: 500,
          textAlign: 'center',
        }}
      >
        Each cluster represents a distinct agent identity. Clear separation enables
        reliable verification with an AUC of 0.992.
      </motion.p>
    </div>
  );
}
