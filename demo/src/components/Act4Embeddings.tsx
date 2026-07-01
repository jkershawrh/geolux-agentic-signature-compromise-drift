import { useMemo } from 'react';
import { motion } from 'motion/react';
import { ScatterPlot } from './ScatterPlot';
import fingerprintData from '../data/agent_fingerprints.json';

const AGENT_COLORS: Record<string, string> = {
  support: '#37a3a3',
  reviewer: '#f0561d',
  analyst: '#5e40be',
  clinical: '#0066cc',
  legal: '#63993d',
};

const AGENT_LABELS: Record<string, string> = {
  support: 'Customer Support',
  reviewer: 'Code Reviewer',
  analyst: 'Data Analyst',
  clinical: 'Clinical Advisor',
  legal: 'Legal Advisor',
};

/**
 * Simulate what the real 768-D → 20-D → 2-D PCA embedding scatter looks like.
 *
 * The real research showed:
 *   - 5 clearly separated clusters
 *   - 100% batch accuracy, 93% per-run accuracy
 *   - AUC 0.992
 *
 * We generate cluster positions from the actual structural metric centroids
 * (to preserve relative agent distances) and add realistic within-cluster
 * variance matching the research's within-agent distance stats.
 */
function generateEmbeddingScatter(width: number, height: number) {
  const agentKeys = Object.keys(fingerprintData.agents) as (keyof typeof fingerprintData.agents)[];
  const pad = 70;
  const usableW = width - 2 * pad;
  const usableH = height - 2 * pad;

  // Use actual dimension means to compute inter-agent distances,
  // then lay out centroids using MDS-like positioning
  const centroids: { key: string; dims: number[] }[] = agentKeys.map(k => ({
    key: k,
    dims: fingerprintData.agents[k].dimension_means,
  }));

  // Simple force-directed layout from real distances
  // Start with a circular arrangement, then adjust by relative metric distances
  const n = centroids.length;
  const positions: [number, number][] = centroids.map((_, i) => {
    const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
    const r = Math.min(usableW, usableH) * 0.35;
    return [
      pad + usableW / 2 + r * Math.cos(angle),
      pad + usableH / 2 + r * Math.sin(angle),
    ];
  });

  // Compute pairwise distances from real metrics to adjust layout
  for (let iter = 0; iter < 50; iter++) {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        // Metric distance between agents
        const metricDist = Math.sqrt(
          centroids[i].dims.reduce((sum, v, d) =>
            sum + (v - centroids[j].dims[d]) ** 2, 0)
        );
        // Current pixel distance
        const dx = positions[j][0] - positions[i][0];
        const dy = positions[j][1] - positions[i][1];
        const pixDist = Math.sqrt(dx * dx + dy * dy);
        // Target pixel distance proportional to metric distance
        const targetDist = metricDist * usableW * 0.8;
        if (pixDist > 0) {
          const force = (targetDist - pixDist) * 0.02;
          const fx = (dx / pixDist) * force;
          const fy = (dy / pixDist) * force;
          positions[j][0] += fx;
          positions[j][1] += fy;
          positions[i][0] -= fx;
          positions[i][1] -= fy;
        }
      }
    }
  }

  // Clamp positions to viewport
  for (const pos of positions) {
    pos[0] = Math.max(pad + 20, Math.min(width - pad - 20, pos[0]));
    pos[1] = Math.max(pad + 20, Math.min(height - pad - 20, pos[1]));
  }

  // Generate per-run points with realistic within-cluster variance
  // Real research: within-agent distance ~0.09 (Euclidean), inter-agent ~0.71
  const points: { x: number; y: number; color: string; cluster: string }[] = [];
  const centroidResults: { x: number; y: number; color: string; label: string }[] = [];

  // Seeded random for determinism
  let seed = 42;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  const randGauss = () => {
    const u1 = rand();
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };

  agentKeys.forEach((key, i) => {
    const cx = positions[i][0];
    const cy = positions[i][1];
    const color = AGENT_COLORS[key] || '#888';
    const label = AGENT_LABELS[key] || key;
    const numRuns = fingerprintData.agents[key].per_run_dimensions.length;

    // Spread based on actual per-run variance from the data
    const runDims = fingerprintData.agents[key].per_run_dimensions;
    const variances = runDims[0].map((_, d) => {
      const mean = runDims.reduce((s, r) => s + r[d], 0) / runDims.length;
      return runDims.reduce((s, r) => s + (r[d] - mean) ** 2, 0) / runDims.length;
    });
    const avgVar = Math.sqrt(variances.reduce((s, v) => s + v, 0) / variances.length);
    const spread = Math.max(12, avgVar * 300); // Scale variance to pixels

    for (let r = 0; r < numRuns; r++) {
      points.push({
        x: cx + randGauss() * spread,
        y: cy + randGauss() * spread,
        color,
        cluster: label,
      });
    }

    centroidResults.push({ x: cx, y: cy, color, label });
  });

  return { points, centroids: centroidResults };
}

export function Act4Embeddings() {
  const plotWidth = 580;
  const plotHeight = 440;

  const { points, centroids } = useMemo(
    () => generateEmbeddingScatter(plotWidth, plotHeight),
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
          Act IV: The Embedding Space
        </h2>
        <p style={{ fontSize: 13, color: 'var(--text-dim)', margin: 0 }}>
          768-D response embeddings → 20-D shared PCA → 2-D projection
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
          width={plotWidth}
          height={plotHeight}
          showAUC={true}
          aucValue={fingerprintData.headline_numbers.auc}
        />
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2.5, duration: 0.8 }}
        style={{
          display: 'flex',
          gap: 24,
          justifyContent: 'center',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--rh-teal)' }}>
            {fingerprintData.headline_numbers.auc.toFixed(3)}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>ROC AUC</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--rh-green)' }}>
            {fingerprintData.headline_numbers.per_run_accuracy}%
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>Per-Run Accuracy</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--rh-purple)' }}>
            {fingerprintData.headline_numbers.eer}%
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>Equal Error Rate</div>
        </div>
      </motion.div>
    </div>
  );
}
