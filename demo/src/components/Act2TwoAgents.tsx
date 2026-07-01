import { useMemo } from 'react';
import { motion } from 'motion/react';
import { RadarChart } from './RadarChart';
import fingerprintData from '../data/agent_fingerprints.json';

function scaleDims(dims: number[]): number[] {
  const allAgents = Object.values(fingerprintData.agents);
  const allDims = allAgents.map(a => a.dimension_means);
  const maxPerDim = dims.map((_, i) => Math.max(...allDims.map(d => d[i]), 0.01));
  return dims.map((v, i) => Math.min(v / maxPerDim[i], 1.0));
}

function scaleRuns(runs: number[][]): number[][] {
  const allAgents = Object.values(fingerprintData.agents);
  const allDims = allAgents.map(a => a.dimension_means);
  const maxPerDim = runs[0].map((_, i) => Math.max(...allDims.map(d => d[i]), 0.01));
  return runs.map(run => run.map((v, i) => Math.min(v / maxPerDim[i], 1.0)));
}

export function Act2TwoAgents() {
  const support = fingerprintData.agents.support;
  const reviewer = fingerprintData.agents.reviewer;
  const dimensionLabels = fingerprintData.dimension_labels;
  const headline = fingerprintData.headline_numbers;
  const scaledSupport = scaleDims(support.dimension_means);
  const scaledReviewer = scaleDims(reviewer.dimension_means);
  const scaledSupportRuns = scaleRuns(support.per_run_dimensions);
  const scaledReviewerRuns = scaleRuns(reviewer.per_run_dimensions);

  // Compute per-dimension deltas and find the most different dimensions
  const dimensionDeltas = useMemo(() => {
    return support.dimension_means.map((v, i) => ({
      label: dimensionLabels[i].replace('\n', ' '),
      delta: Math.abs(v - reviewer.dimension_means[i]),
      supportVal: v,
      reviewerVal: reviewer.dimension_means[i],
    }));
  }, []);

  const topDiffs = useMemo(() => {
    return [...dimensionDeltas]
      .sort((a, b) => b.delta - a.delta)
      .slice(0, 3);
  }, [dimensionDeltas]);

  const stats = [
    { label: 'Fisher Ratio', value: headline.fisher_ratio.toFixed(2), color: 'var(--rh-teal)' },
    { label: "Cohen's d", value: headline.cohens_d.toFixed(2), color: 'var(--rh-orange)' },
    { label: 'p-value', value: '< 0.001', color: 'var(--rh-green)' },
  ];

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
            Alpha &mdash; {support.display_name}
          </p>
          <RadarChart
            values={scaledSupport}
            labels={dimensionLabels}
            color="var(--rh-teal)"
            animate={true}
            size={260}
            varianceData={scaledSupportRuns}
          />
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
            Beta &mdash; {reviewer.display_name}
          </p>
          <RadarChart
            values={scaledReviewer}
            labels={dimensionLabels}
            color="var(--rh-orange)"
            animate={true}
            size={260}
            varianceData={scaledReviewerRuns}
          />
        </motion.div>
      </div>

      {/* Dimension difference table */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1, duration: 0.5 }}
        style={{
          display: 'flex',
          gap: 12,
          flexWrap: 'wrap',
          justifyContent: 'center',
        }}
      >
        {topDiffs.map((d, i) => (
          <div
            key={d.label}
            style={{
              background: 'var(--surface-1)',
              border: '1px solid var(--border)',
              borderRadius: 6,
              padding: '8px 14px',
              textAlign: 'center',
              minWidth: 100,
            }}
          >
            <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 4 }}>
              {d.label}
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center', alignItems: 'baseline' }}>
              <span style={{ fontSize: 13, fontFamily: "'Red Hat Mono', monospace", color: 'var(--rh-teal)' }}>{d.supportVal.toFixed(3)}</span>
              <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>vs</span>
              <span style={{ fontSize: 13, fontFamily: "'Red Hat Mono', monospace", color: 'var(--rh-orange)' }}>{d.reviewerVal.toFixed(3)}</span>
            </div>
            <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--rh-red)', marginTop: 2 }}>
              {'Δ'} {d.delta.toFixed(3)}
            </div>
          </div>
        ))}
      </motion.div>

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
            transition={{ delay: 1.2 + i * 0.2, duration: 0.5 }}
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
