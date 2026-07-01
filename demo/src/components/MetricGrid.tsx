import { motion } from 'motion/react';

interface MetricGridProps {
  metrics: Record<string, number>;
  metricNames: string[];
  delay?: number;
}

function abbreviate(name: string): string {
  return name
    .replace(/_/g, ' ')
    .split(' ')
    .map((w) => w.slice(0, 3))
    .join(' ');
}

function valueColor(v: number): string {
  if (v <= 0) return 'rgba(255,255,255,0.06)';
  if (v < 0.2) return `rgba(55,163,163,${0.2 + v * 2})`;
  if (v < 0.5) return `rgba(55,163,163,${0.3 + v})`;
  if (v < 0.8) return `rgba(55,163,163,${0.5 + v * 0.4})`;
  return `rgba(55,163,163,${0.7 + v * 0.3})`;
}

export function MetricGrid({ metrics, metricNames, delay = 0 }: MetricGridProps) {
  const cols = 6;

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${cols}, 1fr)`,
        gap: 3,
        maxWidth: 400,
        width: '100%',
      }}
    >
      {metricNames.map((name, i) => {
        const value = metrics[name] ?? 0;
        return (
          <motion.div
            key={name}
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: delay + i * 0.015, duration: 0.25 }}
            title={`${name}: ${value.toFixed(4)}`}
            style={{
              position: 'relative',
              background: valueColor(value),
              borderRadius: 3,
              padding: '4px 2px',
              textAlign: 'center',
              cursor: 'default',
              minHeight: 32,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              border: '1px solid rgba(255,255,255,0.04)',
            }}
          >
            <div
              style={{
                fontSize: 7,
                color: 'rgba(255,255,255,0.6)',
                lineHeight: 1.1,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                width: '100%',
              }}
            >
              {abbreviate(name)}
            </div>
            <div
              style={{
                fontSize: 9,
                fontWeight: 700,
                fontFamily: "'Red Hat Mono', monospace",
                color: value > 0 ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.25)',
                marginTop: 1,
              }}
            >
              {value > 0 ? value.toFixed(2) : '--'}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
