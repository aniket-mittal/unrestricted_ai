"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useMemo } from "react";

interface GeneratingIllustrationProps {
  count?: number;
  active?: boolean;
  label?: string;
}

/**
 * Abstract "create_training_pairs" tool-call illustration.
 *
 * A central node (the tool call) emits small sample cards that fan out along
 * three lanes and stream away. No real training text is shown, only abstract
 * units. Lab-instrument feel: hairlines, mechanical ticks, one accent.
 */
export default function GeneratingIllustration({
  count = 6,
  active = false,
  label = "generating samples",
}: GeneratingIllustrationProps) {
  const reduce = useReducedMotion();

  // Clamp to a sensible number of visible sample units.
  const units = Math.max(3, Math.min(count, 9));

  // Three horizontal lanes the samples travel along, fanning from the node.
  const lanes = useMemo(() => [-26, 0, 26], []);

  // Geometry of the viewBox.
  const W = 360;
  const H = 200;
  const nodeX = 96;
  const nodeY = H / 2;
  const exitX = 332;

  // Build the stream of sample cards, distributed across lanes.
  const samples = useMemo(
    () =>
      Array.from({ length: units }, (_, i) => {
        const lane = lanes[i % lanes.length];
        const order = Math.floor(i / lanes.length);
        return { id: i, laneY: nodeY + lane, order };
      }),
    [units, lanes, nodeY],
  );

  const animate = active && !reduce;

  return (
    <figure className="w-full max-w-[360px] select-none">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto"
        role="img"
        aria-label={`${label}: a tool node emitting ${units} sample units`}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        {/* Lane guides: hairline tracks the samples ride along. */}
        <g className="text-border">
          {lanes.map((lane) => (
            <line
              key={lane}
              x1={nodeX + 30}
              y1={nodeY + lane}
              x2={exitX}
              y2={nodeY + lane}
              strokeDasharray="2 5"
              strokeWidth={1}
              opacity={0.7}
            />
          ))}
        </g>

        {/* Exit gate: where samples stream out of frame. */}
        <g className="text-border">
          <line x1={exitX} y1={nodeY - 40} x2={exitX} y2={nodeY + 40} strokeWidth={1} />
          <line x1={exitX - 5} y1={nodeY - 40} x2={exitX} y2={nodeY - 40} strokeWidth={1} />
          <line x1={exitX - 5} y1={nodeY + 40} x2={exitX} y2={nodeY + 40} strokeWidth={1} />
        </g>

        {/* Streaming sample cards. */}
        <g>
          {samples.map((s) => {
            const baseDelay = s.order * 0.42 + (s.laneY === nodeY ? 0 : 0.06);
            const startX = nodeX + 26;
            const endX = exitX - 16;
            return (
              <motion.g
                key={s.id}
                initial={
                  animate
                    ? { opacity: 0, x: 0 }
                    : { opacity: 1, x: startX + (s.order * 56) }
                }
                animate={
                  animate
                    ? {
                        opacity: [0, 1, 1, 0],
                        x: [startX, startX, endX, endX + 18],
                      }
                    : { opacity: 1, x: startX + (s.order * 56) }
                }
                transition={
                  animate
                    ? {
                        duration: 1.7,
                        times: [0, 0.12, 0.85, 1],
                        ease: "easeInOut",
                        repeat: Infinity,
                        repeatDelay: 0.5,
                        delay: baseDelay,
                      }
                    : undefined
                }
              >
                <g transform={`translate(0 ${s.laneY})`}>
                  {/* Sample card: a small abstract unit, no text. */}
                  <rect
                    x={-13}
                    y={-9}
                    width={26}
                    height={18}
                    rx={3}
                    className="text-foreground"
                    fill="hsl(var(--surface))"
                  />
                  {/* Two abstract "lines" of content inside the card. */}
                  <line
                    x1={-8}
                    y1={-3}
                    x2={8}
                    y2={-3}
                    className="text-muted-foreground"
                    strokeWidth={1}
                  />
                  <line
                    x1={-8}
                    y1={3}
                    x2={3}
                    y2={3}
                    className="text-accent"
                    strokeWidth={1.5}
                  />
                </g>
              </motion.g>
            );
          })}
        </g>

        {/* Central tool-call node: the machine spawning samples. */}
        <g transform={`translate(${nodeX} ${nodeY})`}>
          {/* Soft halo that breathes while active. */}
          <motion.circle
            r={30}
            className="text-accent"
            fill="hsl(var(--accent-soft))"
            stroke="none"
            initial={{ opacity: animate ? 0.5 : 0.35, scale: 1 }}
            animate={
              animate
                ? { opacity: [0.35, 0.6, 0.35], scale: [1, 1.06, 1] }
                : { opacity: 0.35, scale: 1 }
            }
            transition={
              animate
                ? { duration: 1.6, ease: "easeInOut", repeat: Infinity }
                : undefined
            }
          />
          {/* Outer ring. */}
          <circle r={26} className="text-border" />
          {/* Inner instrument ring. */}
          <circle r={18} className="text-foreground" />
          {/* Rotating tick that signals work. */}
          <motion.g
            className="text-accent"
            initial={{ rotate: 0 }}
            animate={animate ? { rotate: 360 } : { rotate: 0 }}
            transition={
              animate
                ? { duration: 4, ease: "linear", repeat: Infinity }
                : undefined
            }
            style={{ transformOrigin: "0px 0px" }}
          >
            <line x1={0} y1={-18} x2={0} y2={-12} strokeWidth={2} />
          </motion.g>
          {/* Core glyph: a small "+" denoting creation of a pair. */}
          <g className="text-foreground">
            <line x1={-6} y1={0} x2={6} y2={0} strokeWidth={2} />
            <line x1={0} y1={-6} x2={0} y2={6} strokeWidth={2} />
          </g>
        </g>
      </svg>

      <figcaption className="mt-3 flex items-center gap-2 px-1">
        <span
          className={`inline-block h-1.5 w-1.5 rounded-full ${
            active ? "bg-accent" : "bg-muted-foreground"
          } ${animate ? "animate-pulse-soft" : ""}`}
          aria-hidden="true"
        />
        <span className="font-mono text-xs tracking-tight text-muted-foreground">
          {label}
        </span>
      </figcaption>
    </figure>
  );
}
