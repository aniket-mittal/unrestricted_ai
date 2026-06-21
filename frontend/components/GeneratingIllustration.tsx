"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useMemo } from "react";

interface GeneratingIllustrationProps {
  /** Number of samples being generated (drives the count + density). */
  count?: number;
  /** Whether the animation is actively running. */
  active?: boolean;
  label?: string;
}

const EASE = [0.16, 1, 0.3, 1] as const;

/**
 * "Generating training samples" illustration — matched in polish to the LoRA
 * TrainingIllustration. A teacher node synthesizes DIVERSE prompt/response pairs:
 * mini sample-cards (varied widths = varied phrasing) pop out of the node, stream
 * along three lanes into a collecting stack on the right, while a count ticks up.
 * Card-wrapped, instrument feel, one accent. Reduced-motion shows a static frame.
 */
export default function GeneratingIllustration({
  count = 0,
  active = true,
  label = "Synthesizing diverse pairs",
}: GeneratingIllustrationProps) {
  const reduce = useReducedMotion();
  const animate = active && !reduce;

  // Geometry.
  const W = 320;
  const H = 150;
  const nodeX = 64;
  const nodeY = H / 2;
  const exitX = 250;
  const lanes = useMemo(() => [-34, 0, 34], []);

  // A steady stream of flowing sample cards (visual only; not the real count).
  const FLOW = 6;
  const flow = useMemo(
    () =>
      Array.from({ length: FLOW }, (_, i) => ({
        id: i,
        laneY: nodeY + lanes[i % lanes.length],
        order: Math.floor(i / lanes.length),
        // Varied inner-line widths read as "diverse" prompts/responses.
        wTop: 10 + ((i * 7) % 9),
        wBot: 6 + ((i * 5) % 7),
      })),
    [lanes, nodeY],
  );

  return (
    <div
      className="w-full max-w-[360px] rounded-lg border border-border bg-surface p-5"
      role="group"
      aria-label={`${label}${count ? `, ${count} samples` : ""}`}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            Generating samples
          </p>
          <p className="mt-1 text-sm text-foreground">Teacher writes pairs</p>
        </div>
        <span className="font-mono tnum text-xs text-muted-foreground">
          {count > 0 ? `${count} pairs` : "…"}
        </span>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="mt-3 h-auto w-full"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {/* Lane guides */}
        <g className="text-border">
          {lanes.map((lane) => (
            <line
              key={lane}
              x1={nodeX + 28}
              y1={nodeY + lane}
              x2={exitX - 6}
              y2={nodeY + lane}
              strokeDasharray="2 6"
              strokeWidth={1}
              opacity={0.6}
            />
          ))}
        </g>

        {/* Collecting tray on the right — where pairs pile up. */}
        <g className="text-border">
          <rect
            x={exitX}
            y={nodeY - 44}
            width={56}
            height={88}
            rx={8}
            className="text-border"
            fill="hsl(var(--accent-soft))"
            opacity={0.5}
          />
          <rect x={exitX} y={nodeY - 44} width={56} height={88} rx={8} strokeWidth={1.5} />
          {/* Stacked "collected" bars that fill while active. */}
          {[0, 1, 2, 3].map((i) => (
            <motion.rect
              key={i}
              x={exitX + 10}
              y={nodeY + 30 - i * 17}
              width={36}
              height={10}
              rx={2}
              className="text-foreground"
              fill="hsl(var(--accent))"
              stroke="none"
              initial={{ opacity: animate ? 0 : 0.9, scaleX: animate ? 0.6 : 1 }}
              animate={
                animate
                  ? { opacity: [0, 1, 1], scaleX: [0.6, 1, 1] }
                  : { opacity: 0.9, scaleX: 1 }
              }
              transition={
                animate
                  ? { duration: 2.2, ease: EASE, repeat: Infinity, repeatDelay: 0.4, delay: 0.6 + i * 0.18 }
                  : undefined
              }
              style={{ transformOrigin: `${exitX + 10}px center` }}
            />
          ))}
        </g>

        {/* Streaming sample cards (node -> tray). */}
        {flow.map((s) => {
          const startX = nodeX + 24;
          const endX = exitX - 2;
          const delay = s.order * 0.5 + (s.laneY === nodeY ? 0 : 0.08);
          return (
            <motion.g
              key={s.id}
              initial={
                animate
                  ? { opacity: 0, x: startX, scale: 0.7 }
                  : { opacity: 1, x: startX + s.order * 60, scale: 1 }
              }
              animate={
                animate
                  ? {
                      opacity: [0, 1, 1, 0],
                      x: [startX, startX, endX, endX],
                      scale: [0.7, 1, 1, 0.8],
                    }
                  : { opacity: 1, x: startX + s.order * 60, scale: 1 }
              }
              transition={
                animate
                  ? {
                      duration: 1.9,
                      times: [0, 0.14, 0.82, 1],
                      ease: EASE,
                      repeat: Infinity,
                      repeatDelay: 0.3,
                      delay,
                    }
                  : undefined
              }
            >
              <g transform={`translate(0 ${s.laneY})`}>
                <rect
                  x={-14}
                  y={-10}
                  width={28}
                  height={20}
                  rx={3}
                  className="text-foreground"
                  fill="hsl(var(--surface))"
                />
                {/* varied inner lines => diverse content */}
                <line x1={-9} y1={-3.5} x2={-9 + s.wTop} y2={-3.5} className="text-muted-foreground" strokeWidth={1.2} />
                <line x1={-9} y1={3.5} x2={-9 + s.wBot} y2={3.5} className="text-accent" strokeWidth={1.5} />
              </g>
            </motion.g>
          );
        })}

        {/* Central teacher node. */}
        <g transform={`translate(${nodeX} ${nodeY})`}>
          <motion.circle
            r={28}
            className="text-accent"
            fill="hsl(var(--accent-soft))"
            stroke="none"
            initial={{ opacity: animate ? 0.5 : 0.35, scale: 1 }}
            animate={animate ? { opacity: [0.35, 0.6, 0.35], scale: [1, 1.08, 1] } : { opacity: 0.35, scale: 1 }}
            transition={animate ? { duration: 1.6, ease: "easeInOut", repeat: Infinity } : undefined}
          />
          <circle r={24} className="text-border" />
          <circle r={16} className="text-foreground" />
          {/* Rotating dual ticks signal active synthesis. */}
          <motion.g
            className="text-accent"
            initial={{ rotate: 0 }}
            animate={animate ? { rotate: 360 } : { rotate: 0 }}
            transition={animate ? { duration: 3.2, ease: "linear", repeat: Infinity } : undefined}
            style={{ transformOrigin: "0px 0px" }}
          >
            <line x1={0} y1={-16} x2={0} y2={-10} strokeWidth={2} />
            <line x1={0} y1={16} x2={0} y2={10} strokeWidth={2} />
          </motion.g>
          {/* Core "+" denotes creating a pair. */}
          <g className="text-foreground">
            <line x1={-5} y1={0} x2={5} y2={0} strokeWidth={2} />
            <line x1={0} y1={-5} x2={0} y2={5} strokeWidth={2} />
          </g>
        </g>
      </svg>

      <div className="mt-3 flex items-center gap-2">
        <span
          className={`inline-block h-1.5 w-1.5 rounded-full ${active ? "bg-accent" : "bg-muted-foreground"} ${animate ? "animate-pulse-soft" : ""}`}
          aria-hidden="true"
        />
        <span className="font-mono text-xs tracking-tight text-muted-foreground">{label}</span>
      </div>
    </div>
  );
}
