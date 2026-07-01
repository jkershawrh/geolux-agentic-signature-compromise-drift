import { useRef, useEffect, useCallback } from 'react';

const DIMENSIONS = [
  'Response\nStructure',
  'Token\nEconomics',
  'Tool\nBehavior',
  'Reasoning\nPattern',
  'Temporal\nProfile',
  'Semantic\nConsistency',
  'Safety\nAlignment',
  'Agent\nSpecific',
  'Embedding',
];

interface RadarChartProps {
  values: number[];
  labels?: string[];
  color: string;
  animate?: boolean;
  size?: number;
}

export function RadarChart({
  values,
  labels = DIMENSIONS,
  color,
  animate = true,
  size = 300,
}: RadarChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animProgress = useRef(0);
  const rafRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);

  const draw = useCallback(
    (progress: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const w = size;
      const h = size;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.scale(dpr, dpr);

      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const maxR = Math.min(cx, cy) - 50;
      const n = values.length;
      const angleStep = (Math.PI * 2) / n;
      const startAngle = -Math.PI / 2;

      // Draw grid rings
      const rings = [0.25, 0.5, 0.75, 1.0];
      for (const ring of rings) {
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
          const angle = startAngle + i * angleStep;
          const x = cx + Math.cos(angle) * maxR * ring;
          const y = cy + Math.sin(angle) * maxR * ring;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Draw axis lines
      for (let i = 0; i < n; i++) {
        const angle = startAngle + i * angleStep;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(angle) * maxR, cy + Math.sin(angle) * maxR);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Draw labels
      ctx.font = '10px "Red Hat Text", sans-serif';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.textAlign = 'center';
      for (let i = 0; i < n; i++) {
        const angle = startAngle + i * angleStep;
        const labelR = maxR + 28;
        const x = cx + Math.cos(angle) * labelR;
        const y = cy + Math.sin(angle) * labelR;
        const lines = labels[i].split('\n');
        lines.forEach((line, li) => {
          ctx.fillText(line, x, y + li * 12 - ((lines.length - 1) * 6));
        });
      }

      // Draw data polygon
      const currentValues = values.map((v) => v * progress);
      ctx.beginPath();
      for (let i = 0; i <= n; i++) {
        const idx = i % n;
        const angle = startAngle + idx * angleStep;
        const r = maxR * currentValues[idx];
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();

      // Fill
      ctx.fillStyle = color + '33';
      ctx.fill();

      // Stroke
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw dots
      for (let i = 0; i < n; i++) {
        const angle = startAngle + i * angleStep;
        const r = maxR * currentValues[i];
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    },
    [values, labels, color, size],
  );

  useEffect(() => {
    if (!animate) {
      animProgress.current = 1;
      draw(1);
      return;
    }

    animProgress.current = 0;
    startTimeRef.current = performance.now();

    const tick = (now: number) => {
      const elapsed = now - startTimeRef.current;
      const t = Math.min(elapsed / 1000, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - t, 3);
      animProgress.current = eased;
      draw(eased);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [animate, draw]);

  return <canvas ref={canvasRef} style={{ display: 'block' }} />;
}
