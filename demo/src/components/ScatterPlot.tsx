import { useRef, useEffect } from 'react';

interface Point {
  x: number;
  y: number;
  color: string;
  cluster: string;
}

interface Centroid {
  x: number;
  y: number;
  color: string;
  label: string;
}

interface ScatterPlotProps {
  points: Point[];
  centroids: Centroid[];
  animate?: boolean;
  width?: number;
  height?: number;
  showAUC?: boolean;
  aucValue?: number;
}

/** Compute 2D covariance and draw a confidence ellipse (95%) */
function computeEllipse(pts: { x: number; y: number }[], centroid: { x: number; y: number }) {
  if (pts.length < 3) return null;

  let sxx = 0, syy = 0, sxy = 0;
  for (const p of pts) {
    const dx = p.x - centroid.x;
    const dy = p.y - centroid.y;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  const n = pts.length;
  sxx /= n;
  syy /= n;
  sxy /= n;

  // Eigenvalues
  const trace = sxx + syy;
  const det = sxx * syy - sxy * sxy;
  const disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
  const l1 = trace / 2 + disc;
  const l2 = trace / 2 - disc;

  // Eigenvector angle
  const angle = Math.atan2(2 * sxy, sxx - syy) / 2;

  // 95% confidence scale (chi-squared 2 DOF, p=0.05 => 5.991)
  const scale = Math.sqrt(5.991);

  return {
    rx: scale * Math.sqrt(Math.max(0, l1)),
    ry: scale * Math.sqrt(Math.max(0, l2)),
    angle,
  };
}

export function ScatterPlot({
  points,
  centroids,
  animate = true,
  width = 600,
  height = 450,
  showAUC = false,
  aucValue = 0.992,
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

    // Group points by cluster for ellipse computation
    const clusterPoints: Record<string, { x: number; y: number }[]> = {};
    for (const p of points) {
      if (!clusterPoints[p.cluster]) clusterPoints[p.cluster] = [];
      clusterPoints[p.cluster].push({ x: p.x, y: p.y });
    }

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

      const ellipseProgress = animate ? Math.min(Math.max(progress - 0.3, 0) / 0.7, 1) : 1;

      // Draw confidence ellipses behind everything
      if (ellipseProgress > 0) {
        for (const c of centroids) {
          const pts = clusterPoints[c.label];
          if (!pts) continue;
          const ellipse = computeEllipse(pts, c);
          if (!ellipse) continue;

          ctx.save();
          ctx.translate(c.x, c.y);
          ctx.rotate(ellipse.angle);
          ctx.globalAlpha = 0.08 * ellipseProgress;
          ctx.beginPath();
          ctx.ellipse(0, 0, ellipse.rx * ellipseProgress, ellipse.ry * ellipseProgress, 0, 0, Math.PI * 2);
          ctx.fillStyle = c.color;
          ctx.fill();
          ctx.globalAlpha = 0.25 * ellipseProgress;
          ctx.strokeStyle = c.color;
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.restore();
        }
      }

      // Draw inter-cluster distance lines (faint)
      if (ellipseProgress > 0.5) {
        const distAlpha = Math.min((ellipseProgress - 0.5) / 0.5, 1) * 0.15;
        for (let i = 0; i < centroids.length; i++) {
          for (let j = i + 1; j < centroids.length; j++) {
            const a = centroids[i];
            const b = centroids[j];
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = `rgba(255,255,255,${distAlpha})`;
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 6]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Distance label at midpoint
            const mx = (a.x + b.x) / 2;
            const my = (a.y + b.y) / 2;
            ctx.font = '9px "Red Hat Mono", monospace';
            ctx.fillStyle = `rgba(255,255,255,${distAlpha * 2})`;
            ctx.textAlign = 'center';
            ctx.fillText(dist.toFixed(0), mx, my - 4);
          }
        }
      }

      // Draw lines from points to centroids
      for (let i = 0; i < visibleCount; i++) {
        const p = points[i];
        const centroid = centroids.find((c) => c.label === p.cluster);
        if (centroid) {
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(centroid.x, centroid.y);
          ctx.strokeStyle = p.color + '18';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // Draw points with a small bounce effect
      for (let i = 0; i < visibleCount; i++) {
        const p = points[i];
        let pointScale = 1;
        if (animate) {
          const pointProgress = (visibleCount - i) / Math.max(points.length * 0.1, 1);
          if (pointProgress < 1 && pointProgress > 0) {
            // Small bounce for recently appeared points
            const t = pointProgress;
            pointScale = 1 + Math.sin(t * Math.PI) * 0.5;
          }
        }
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4 * pointScale, 0, Math.PI * 2);
        ctx.fillStyle = p.color + 'cc';
        ctx.fill();
        ctx.strokeStyle = p.color;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Draw centroids
      const centroidProgress = animate ? Math.min(progress * 1.5, 1) : 1;
      for (const c of centroids) {
        const cSize = 6 * centroidProgress;
        ctx.beginPath();
        ctx.arc(c.x, c.y, cSize, 0, Math.PI * 2);
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
        ctx.fillText(`AUC: ${aucValue.toFixed(3)}`, width - 20, 30);
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
  }, [points, centroids, animate, width, height, showAUC, aucValue]);

  return <canvas ref={canvasRef} style={{ display: 'block' }} />;
}
