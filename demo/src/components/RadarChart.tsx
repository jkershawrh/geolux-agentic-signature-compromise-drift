import { useRef, useEffect } from 'react';

interface RadarChartProps {
  values: number[];
  labels?: string[];
  color: string;
  animate?: boolean;
  size?: number;
  varianceData?: number[][];
  metricDots?: { dimIndex: number; value: number }[];
}

export function RadarChart({
  values,
  labels,
  color,
  animate = true,
  size = 300,
  varianceData,
  metricDots,
}: RadarChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;

    const cx = size / 2;
    const cy = size / 2;
    const maxR = Math.min(cx, cy) - 50;
    const n = values.length;
    if (n === 0) return;
    const angleStep = (Math.PI * 2) / n;
    const startAngle = -Math.PI / 2;
    const defaultLabels = Array.from({ length: n }, (_, i) => `Dim ${i}`);
    const useLabels = labels || defaultLabels;

    const startTime = performance.now();

    function render(progress: number) {
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx!.clearRect(0, 0, size, size);

      // Grid rings
      for (const ring of [0.25, 0.5, 0.75, 1.0]) {
        ctx!.beginPath();
        for (let i = 0; i <= n; i++) {
          const a = startAngle + (i % n) * angleStep;
          const x = cx + Math.cos(a) * maxR * ring;
          const y = cy + Math.sin(a) * maxR * ring;
          i === 0 ? ctx!.moveTo(x, y) : ctx!.lineTo(x, y);
        }
        ctx!.closePath();
        ctx!.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx!.lineWidth = 1;
        ctx!.stroke();
      }

      // Axis lines
      for (let i = 0; i < n; i++) {
        const a = startAngle + i * angleStep;
        ctx!.beginPath();
        ctx!.moveTo(cx, cy);
        ctx!.lineTo(cx + Math.cos(a) * maxR, cy + Math.sin(a) * maxR);
        ctx!.strokeStyle = 'rgba(255, 255, 255, 0.06)';
        ctx!.lineWidth = 1;
        ctx!.stroke();
      }

      // Labels
      ctx!.font = '10px "Red Hat Text", sans-serif';
      ctx!.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx!.textAlign = 'center';
      for (let i = 0; i < n; i++) {
        const a = startAngle + i * angleStep;
        const lr = maxR + 28;
        const x = cx + Math.cos(a) * lr;
        const y = cy + Math.sin(a) * lr;
        const lines = useLabels[i].split('\\n');
        lines.forEach((line, li) => {
          ctx!.fillText(line, x, y + li * 12 - ((lines.length - 1) * 6));
        });
      }

      // Variance bands
      if (varianceData && varianceData.length > 0 && progress > 0) {
        const mins = new Array(n).fill(Infinity);
        const maxes = new Array(n).fill(-Infinity);
        for (const run of varianceData) {
          for (let i = 0; i < n; i++) {
            if (run[i] < mins[i]) mins[i] = run[i];
            if (run[i] > maxes[i]) maxes[i] = run[i];
          }
        }
        ctx!.save();
        ctx!.globalAlpha = 0.12 * progress;
        ctx!.beginPath();
        for (let i = 0; i <= n; i++) {
          const idx = i % n;
          const a = startAngle + idx * angleStep;
          const x = cx + Math.cos(a) * maxR * maxes[idx] * progress;
          const y = cy + Math.sin(a) * maxR * maxes[idx] * progress;
          i === 0 ? ctx!.moveTo(x, y) : ctx!.lineTo(x, y);
        }
        for (let i = n; i >= 0; i--) {
          const idx = i % n;
          const a = startAngle + idx * angleStep;
          const x = cx + Math.cos(a) * maxR * mins[idx] * progress;
          const y = cy + Math.sin(a) * maxR * mins[idx] * progress;
          ctx!.lineTo(x, y);
        }
        ctx!.closePath();
        ctx!.fillStyle = color;
        ctx!.fill();
        ctx!.restore();
      }

      // Data polygon with glow
      const cv = values.map(v => v * progress);

      ctx!.save();
      ctx!.shadowColor = color;
      ctx!.shadowBlur = 12;
      ctx!.beginPath();
      for (let i = 0; i <= n; i++) {
        const idx = i % n;
        const a = startAngle + idx * angleStep;
        const x = cx + Math.cos(a) * maxR * cv[idx];
        const y = cy + Math.sin(a) * maxR * cv[idx];
        i === 0 ? ctx!.moveTo(x, y) : ctx!.lineTo(x, y);
      }
      ctx!.closePath();
      ctx!.fillStyle = color + '33';
      ctx!.fill();
      ctx!.strokeStyle = color;
      ctx!.lineWidth = 2;
      ctx!.stroke();
      ctx!.restore();

      // Vertex dots
      for (let i = 0; i < n; i++) {
        const a = startAngle + i * angleStep;
        const x = cx + Math.cos(a) * maxR * cv[i];
        const y = cy + Math.sin(a) * maxR * cv[i];
        ctx!.beginPath();
        ctx!.arc(x, y, 3, 0, Math.PI * 2);
        ctx!.fillStyle = color;
        ctx!.fill();
      }

      // Metric dots
      if (metricDots && progress > 0.5) {
        ctx!.globalAlpha = Math.min((progress - 0.5) / 0.5, 1) * 0.5;
        for (const dot of metricDots) {
          const a = startAngle + dot.dimIndex * angleStep;
          const x = cx + Math.cos(a) * maxR * dot.value * progress;
          const y = cy + Math.sin(a) * maxR * dot.value * progress;
          ctx!.beginPath();
          ctx!.arc(x, y, 1.5, 0, Math.PI * 2);
          ctx!.fillStyle = color;
          ctx!.fill();
        }
        ctx!.globalAlpha = 1;
      }
    }

    if (!animate) {
      render(1);
      return;
    }

    const tick = () => {
      const elapsed = performance.now() - startTime;
      const t = Math.min(elapsed / 1000, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      render(eased);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [values, labels, color, animate, size, varianceData, metricDots]);

  return <canvas ref={canvasRef} style={{ display: 'block' }} />;
}
