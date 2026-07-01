import { useMemo } from 'react';
import { motion } from 'motion/react';
import { RadarChart } from './RadarChart';
import { MetricGrid } from './MetricGrid';
import fingerprintData from '../data/agent_fingerprints.json';

/** Scale dimension values relative to the max across all agents so shapes fill the chart */
function scaleDimensions(dims: number[]): number[] {
  const allAgents = Object.values(fingerprintData.agents);
  const allDims = allAgents.map(a => a.dimension_means);
  const maxPerDim = dims.map((_, i) => Math.max(...allDims.map(d => d[i]), 0.01));
  return dims.map((v, i) => Math.min(v / maxPerDim[i], 1.0));
}

function scaleRunDimensions(runs: number[][]): number[][] {
  const allAgents = Object.values(fingerprintData.agents);
  const allDims = allAgents.map(a => a.dimension_means);
  const maxPerDim = runs[0].map((_, i) => Math.max(...allDims.map(d => d[i]), 0.01));
  return runs.map(run => run.map((v, i) => Math.min(v / maxPerDim[i], 1.0)));
}

export function Act1Fingerprint() {
  const support = fingerprintData.agents.support;
  const dimensionLabels = fingerprintData.dimension_labels;
  const dimensionSizes = fingerprintData.dimension_sizes;
  const metricNames = fingerprintData.metric_names;
  const scaledDims = scaleDimensions(support.dimension_means);
  const scaledRuns = scaleRunDimensions(support.per_run_dimensions);

  // Build metric dots: map each metric to its dimension axis
  const metricDots = useMemo(() => {
    const dots: { dimIndex: number; value: number }[] = [];
    let offset = 0;
    for (let d = 0; d < dimensionSizes.length; d++) {
      for (let m = 0; m < dimensionSizes[d]; m++) {
        const metricName = metricNames[offset + m];
        const value = support.metric_means[metricName as keyof typeof support.metric_means] ?? 0;
        if (value > 0) {
          dots.push({ dimIndex: d, value });
        }
      }
      offset += dimensionSizes[d];
    }
    return dots;
  }, []);

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px 24px',
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
          Act I: The Fingerprint
        </h2>
        <p style={{ fontSize: 13, color: 'var(--text-dim)', margin: 0 }}>
          Agent Alpha &mdash; {support.display_name}
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3, duration: 0.6 }}
      >
        <RadarChart
          values={scaledDims}
          labels={dimensionLabels}
          color="var(--rh-teal)"
          animate={true}
          size={320}
          varianceData={scaledRuns}
          metricDots={metricDots}
        />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.2, duration: 0.6 }}
      >
        <div style={{ fontSize: 11, color: 'var(--text-dim)', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 6, textAlign: 'center' }}>
          36 Metric Fingerprint
        </div>
        <MetricGrid metrics={support.metric_means} metricNames={metricNames} delay={1.4} />
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
          `Every agent develops a unique behavioral fingerprint across ${dimensionLabels.length} dimensions.`,
          `These dimensions capture how an agent structures responses, uses tokens, and reasons through problems.`,
          `${metricNames.length} individual metrics are extracted per interaction, then projected into a dense embedding.`,
        ].map((text, i) => (
          <motion.p
            key={i}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 1.8 + i * 0.3, duration: 0.5 }}
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
