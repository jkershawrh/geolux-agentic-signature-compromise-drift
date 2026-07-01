import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { RadarChart } from './RadarChart';
import fingerprintData from '../data/agent_fingerprints.json';

const support = fingerprintData.agents.support;
const reviewer = fingerprintData.agents.reviewer;
const dimensionLabels = fingerprintData.dimension_labels;
const headline = fingerprintData.headline_numbers;

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

const ALPHA_VALUES = scaleDims(support.dimension_means);
const REVIEWER_SCALED = scaleDims(reviewer.dimension_means);

function generateImpostorValues(base: number[], noise: number[]): number[] {
  return base.map((v, i) => {
    const reviewerLeak = (noise[i] - v) * (0.15 + Math.random() * 0.35);
    const jitter = (Math.random() - 0.5) * 0.12;
    return Math.max(0, Math.min(1, v + reviewerLeak + jitter));
  });
}

function computeDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum / a.length);
}

export function Act3Impostor() {
  const [impostorValues, setImpostorValues] = useState(() =>
    generateImpostorValues(ALPHA_VALUES, REVIEWER_SCALED),
  );
  const [distance, setDistance] = useState(() =>
    computeDistance(ALPHA_VALUES, impostorValues),
  );

  useEffect(() => {
    const interval = setInterval(() => {
      const newValues = generateImpostorValues(ALPHA_VALUES, reviewer.dimension_means);
      setImpostorValues(newValues);
      setDistance(computeDistance(ALPHA_VALUES, newValues));
    }, 1500);
    return () => clearInterval(interval);
  }, []);

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
          Act III: The Impostor
        </h2>
        <p style={{ fontSize: 13, color: 'var(--text-dim)', margin: 0 }}>
          Can a compromised agent mimic the real one?
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
            Genuine Alpha
          </p>
          <RadarChart
            values={ALPHA_VALUES}
            labels={dimensionLabels}
            color="var(--rh-teal)"
            animate={false}
            size={260}
            varianceData={scaleRuns(support.per_run_dimensions)}
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
              color: 'var(--rh-red)',
              letterSpacing: 2,
              margin: '0 0 8px',
              textTransform: 'uppercase',
            }}
          >
            Impostor &mdash; Morphing
          </p>
          <RadarChart
            values={impostorValues}
            labels={dimensionLabels}
            color="var(--rh-red)"
            animate={false}
            size={260}
          />
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6, duration: 0.5 }}
        style={{
          background: 'var(--surface-1)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: '16px 32px',
          textAlign: 'center',
        }}
      >
        <div
          style={{
            fontSize: 12,
            color: 'var(--text-dim)',
            letterSpacing: 1,
            textTransform: 'uppercase',
            marginBottom: 8,
          }}
        >
          Euclidean Distance (9-dim)
        </div>
        <div
          style={{
            fontSize: 32,
            fontWeight: 700,
            fontFamily: "'Red Hat Mono', monospace",
            color: distance > 0.15 ? 'var(--rh-red)' : 'var(--rh-orange)',
            transition: 'color 0.3s',
          }}
        >
          {distance.toFixed(4)}
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 1, duration: 0.6 }}
        style={{
          background: 'rgba(238, 0, 0, 0.08)',
          border: '1px solid rgba(238, 0, 0, 0.3)',
          borderRadius: 8,
          padding: '16px 32px',
          textAlign: 'center',
          maxWidth: 400,
        }}
      >
        <div
          style={{
            fontSize: 28,
            fontWeight: 800,
            fontFamily: "'Red Hat Display', sans-serif",
            color: 'var(--rh-red)',
          }}
        >
          EER: {headline.eer}% {'±'} {headline.eer_ci}%
        </div>
        <div
          style={{
            fontSize: 13,
            color: 'var(--text-secondary)',
            marginTop: 4,
          }}
        >
          {(100 - headline.eer).toFixed(1)}% of impostors detected
        </div>
      </motion.div>
    </div>
  );
}
