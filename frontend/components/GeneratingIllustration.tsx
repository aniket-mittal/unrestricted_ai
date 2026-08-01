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

  // Geometry. Height is tuned to vertically match the LoRA TrainingIllustration.
  //
  // Spacing rules that keep this from looking stretched:
  //   - cards must clear the node's outer glow (r=28) before they appear, or
  //     they overlap the "+" and the whole thing reads as a collision;
  //   - lanes stay well inside the box so the outer rows are not crowding the
  //     top and bottom edges;
  //   - the tray is sized to hold its four bars with even margins.
  const W = 320;
  const H = 150;
  const nodeX = 52;
  const nodeY = H / 2;
  const nodeR = 26; // outer glow radius
  const trayX = 238;
  const trayW = 60;
  const trayH = 88;
  const lanes = useMemo(() => [-30, 0, 30], []);
  // Cards travel from just clear of the node to just inside the tray.
  const startX = nodeX + nodeR + 12;
  const endX = trayX - 10;

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
    /* No border/background here: this renders INSIDE ActivityCard, which is
       already a bordered surface. A second frame made the graphic look boxed in. */
    <div
      className="w-full"
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
        className="my-4 h-auto w-full"
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
              x1={startX - 6}
              y1={nodeY + lane}
              x2={endX + 2}
              y2={nodeY + lane}
              strokeDasharray="2 6"
              strokeWidth={1}
              opacity={0.55}
            />
          ))}
        </g>

        {/* Collecting tray on the right — where pairs pile up. */}
        <g className="text-border">
          <rect
            x={trayX}
            y={nodeY - trayH / 2}
            width={trayW}
            height={trayH}
            rx={8}
            className="text-border"
            fill="hsl(var(--accent-soft))"
            opacity={0.5}
          />
          <rect x={trayX} y={nodeY - trayH / 2} width={trayW} height={trayH} rx={8} strokeWidth={1.5} />
          {/* Bars fill bottom-up as pairs land, then the whole stack fades and
              the cycle restarts. Each bar SLIDES IN from the left and settles,
              rather than just scaling in place, so it reads as a pair arriving
              from the lanes rather than a loading gauge. */}
          {[0, 1, 2, 3].map((i) => {
            const barY = nodeY + trayH / 2 - 16 - i * 15;
            return (
              <motion.rect
                key={i}
                x={trayX + 9}
                y={barY}
                width={trayW - 18}
                height={9}
                rx={2}
                fill="hsl(var(--accent))"
                stroke="none"
                initial={{ opacity: animate ? 0 : 0.9, x: 0 }}
                animate={
                  animate
                    ? { opacity: [0, 1, 1, 1, 0], x: [-14, 0, 0, 0, 0] }
                    : { opacity: 0.9, x: 0 }
                }
                transition={
                  animate
                    ? {
                        duration: 3.4,
                        // Land quickly, hold while the rest arrive, fade together.
                        times: [0, 0.12, 0.3, 0.86, 1],
                        ease: EASE,
                        repeat: Infinity,
                        delay: i * 0.34,
                      }
                    : undefined
                }
              />
            );
          })}
        </g>

        {/* Streaming sample cards (node -> tray). */}
        {flow.map((s) => {
          // Stagger every card across the first half of the cycle so the lanes
          // stay populated instead of emitting in two visible clumps.
          const delay = s.id * 0.28;
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
                      // Trailing 0-opacity keyframe parks the card off-stage for
                      // the remainder of the shared cycle.
                      opacity: [0, 1, 1, 0, 0],
                      x: [startX, startX, endX, endX, startX],
                      scale: [0.7, 1, 1, 0.8, 0.7],
                    }
                  : { opacity: 1, x: startX + s.order * 60, scale: 1 }
              }
              transition={
                animate
                  ? {
                      // Same 3.4s cycle as the tray bars, so a card reaching the
                      // tray coincides with a bar landing instead of the two
                      // loops drifting against each other.
                      duration: 3.4,
                      times: [0, 0.08, 0.42, 0.5, 1],
                      ease: EASE,
                      repeat: Infinity,
                      delay,
                    }
                  : undefined
              }
            >
              <g transform={`translate(0 ${s.laneY})`}>
                <rect
                  x={-13}
                  y={-9}
                  width={26}
                  height={18}
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
            r={nodeR}
            className="text-accent"
            fill="hsl(var(--accent-soft))"
            stroke="none"
            initial={{ opacity: animate ? 0.5 : 0.35, scale: 1 }}
            animate={animate ? { opacity: [0.35, 0.6, 0.35], scale: [1, 1.08, 1] } : { opacity: 0.35, scale: 1 }}
            transition={animate ? { duration: 1.6, ease: "easeInOut", repeat: Infinity } : undefined}
          />
          <circle r={22} className="text-border" />
          <circle r={15} className="text-foreground" />
          {/* Rotating dual ticks signal active synthesis. */}
          <motion.g
            className="text-accent"
            initial={{ rotate: 0 }}
            animate={animate ? { rotate: 360 } : { rotate: 0 }}
            transition={animate ? { duration: 3.2, ease: "linear", repeat: Infinity } : undefined}
            style={{ transformOrigin: "0px 0px" }}
          >
            <line x1={0} y1={-15} x2={0} y2={-9} strokeWidth={2} />
            <line x1={0} y1={15} x2={0} y2={9} strokeWidth={2} />
          </motion.g>
          {/* Core "+" denotes creating a pair. */}
          <g className="text-foreground">
            <line x1={-5} y1={0} x2={5} y2={0} strokeWidth={2} />
            <line x1={0} y1={-5} x2={0} y2={5} strokeWidth={2} />
          </g>
        </g>
      </svg>

      <div className="mt-1 flex items-center gap-2">
        <span
          className={`inline-block h-1.5 w-1.5 rounded-full ${active ? "bg-accent" : "bg-muted-foreground"} ${animate ? "animate-pulse-soft" : ""}`}
          aria-hidden="true"
        />
        <span className="font-mono text-xs tracking-tight text-muted-foreground">{label}</span>
      </div>
    </div>
  );
}
