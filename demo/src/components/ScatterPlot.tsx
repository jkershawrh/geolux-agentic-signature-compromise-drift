import { useRef, useEffect } from 'react';

interface Point {
  x: number;
  y: number;
  color: string;
  cluster: string;
}

interface ScatterPlotProps {
  points: Point[];
  centroids: { x: number; y: number; color: string; label: string }[];
  animate?: boolean;
  width?: number;
  height?: number;
  showAUC?: boolean;
}

export function ScatterPlot({
  points,
  centroids,
  animate = true,
  width = 600,
  height = 450,
  showAUC = false,
}: ScatterPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    startTimeRef.current = performance.now();

    const draw = (progress: number) => {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      // Background grid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;
      for (let x = 0; x <= width; x += 50) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y <= height; y += 50) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      const visibleCount = animate
        ? Math.floor(points.length * Math.min(progress, 1))
        : points.length;

      // Draw lines to centroids
      for (let i = 0; i < visibleCount; i++) {
        const p = points[i];
        const centroid = centroids.find((c) => c.label === p.cluster);
        if (centroid) {
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(centroid.x, centroid.y);
          ctx.strokeStyle = p.color + '20';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // Draw points
      for (let i = 0; i < visibleCount; i++) {
        const p = points[i];
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = p.color + 'cc';
        ctx.fill();
        ctx.strokeStyle = p.color;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Draw centroids
      const centroidProgress = animate ? Math.min(progress * 1.5, 1) : 1;
      for (const c of centroids) {
        const size = 6 * centroidProgress;
        ctx.beginPath();
        ctx.arc(c.x, c.y, size, 0, Math.PI * 2);
        ctx.fillStyle = c.color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        if (centroidProgress > 0.5) {
          ctx.font = '11px "Red Hat Text", sans-serif';
          ctx.fillStyle = c.color;
          ctx.textAlign = 'center';
          ctx.fillText(c.label, c.x, c.y - 14);
        }
      }

      // AUC label
      if (showAUC && progress > 0.8) {
        const alpha = Math.min((progress - 0.8) / 0.2, 1);
        ctx.font = 'bold 16px "Red Hat Display", sans-serif';
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
        ctx.textAlign = 'right';
        ctx.fillText('AUC: 0.992', width - 20, 30);
      }
    };

    if (!animate) {
      draw(1);
      return;
    }

    const tick = (now: number) => {
      const elapsed = now - startTimeRef.current;
      const progress = elapsed / 2500;
      draw(progress);
      if (progress < 1.2) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [points, centroids, animate, width, height, showAUC]);

  return <canvas ref={canvasRef} style={{ display: 'block' }} />;
}
