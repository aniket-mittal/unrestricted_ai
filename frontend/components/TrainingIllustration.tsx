"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";

export interface TrainingIllustrationProps {
  step: number;
  totalSteps: number;
  loss: number;
  version?: string;
  phase: "training" | "done";
}

// Geometry for the progress arc. A 270deg sweep, instrument-style.
const SIZE = 168;
const CENTER = SIZE / 2;
const RADIUS = 64;
const STROKE = 6;
const START_ANGLE = 135; // bottom-left
const SWEEP = 270; // degrees
const CIRC = (2 * Math.PI * RADIUS * SWEEP) / 360;

// Weights-settling grid.
const GRID = 6; // 6x6 cells

function polar(angleDeg: number): { x: number; y: number } {
  const a = (angleDeg * Math.PI) / 180;
  return { x: CENTER + RADIUS * Math.cos(a), y: CENTER + RADIUS * Math.sin(a) };
}

// Smoothly tween a displayed number toward a target so the loss readout
// "ticks" rather than jumping. Skipped under reduced motion.
function useTickingNumber(target: number, reduced: boolean): number {
  const [value, setValue] = useState(target);
  const frame = useRef<number | null>(null);
  const fromRef = useRef(target);
  const startRef = useRef(0);

  useEffect(() => {
    if (reduced) {
      setValue(target);
      return;
    }
    fromRef.current = value;
    startRef.current = performance.now();
    const from = fromRef.current;
    const dur = 280;

    const tick = (now: number) => {
      const t = Math.min(1, (now - startRef.current) / dur);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      setValue(from + (target - from) * eased);
      if (t < 1) {
        frame.current = requestAnimationFrame(tick);
      }
    };
    frame.current = requestAnimationFrame(tick);
    return () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, reduced]);

  return reduced ? target : value;
}

export default function TrainingIllustration({
  step,
  totalSteps,
  loss,
  version,
  phase,
}: TrainingIllustrationProps): JSX.Element {
  const reduced = useReducedMotion() ?? false;

  const safeTotal = totalSteps > 0 ? totalSteps : 1;
  const ratio = phase === "done" ? 1 : Math.min(1, Math.max(0, step / safeTotal));
  const pct = Math.round(ratio * 100);

  const displayLoss = useTickingNumber(loss, reduced);

  const startPoint = polar(START_ANGLE);
  const arcLength = CIRC * ratio;

  // How many grid cells are "locked in" given current progress.
  const totalCells = GRID * GRID;
  const lockedCells = phase === "done" ? totalCells : Math.round(ratio * totalCells);

  const enter = { duration: reduced ? 0 : 0.24, ease: [0.16, 1, 0.3, 1] as const };

  return (
    /* Unframed: ActivityCard already provides the border and surface. */
    <div
      className="w-full"
      role="group"
      aria-label={
        phase === "done"
          ? `Training complete${version ? `, ${version}` : ""}`
          : `Training in progress, ${pct} percent, ${step} of ${safeTotal} steps`
      }
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            {phase === "done" ? "Adapter merged" : "Training adapter"}
          </p>
          <p className="mt-1 text-sm text-foreground">
            {phase === "done" ? "Weights settled" : "LoRA fine-tune"}
          </p>
        </div>
        <span
          className="font-mono tnum text-xs text-muted-foreground"
          aria-hidden={phase === "done"}
        >
          {phase === "done" ? "" : `${step}/${safeTotal}`}
        </span>
      </div>

      <div className="mt-4 flex items-center gap-5">
        {/* Instrument arc */}
        <div className="relative shrink-0" style={{ width: SIZE / 1.4, height: SIZE / 1.4 }}>
          <svg
            viewBox={`0 0 ${SIZE} ${SIZE}`}
            width="100%"
            height="100%"
            aria-hidden="true"
          >
            {/* Track */}
            <path
              d={describeArc(START_ANGLE, START_ANGLE + SWEEP)}
              fill="none"
              stroke="currentColor"
              className="text-border"
              strokeWidth={STROKE}
              strokeLinecap="round"
            />
            {/* Filled progress */}
            <motion.path
              d={describeArc(START_ANGLE, START_ANGLE + SWEEP)}
              fill="none"
              stroke="currentColor"
              className="text-accent"
              strokeWidth={STROKE}
              strokeLinecap="round"
              strokeDasharray={CIRC}
              initial={false}
              animate={{ strokeDashoffset: CIRC - arcLength }}
              transition={{ duration: reduced ? 0 : 0.3, ease: [0.16, 1, 0.3, 1] }}
            />
            {/* Tip marker */}
            {!reduced && phase === "training" && (
              <motion.circle
                r={STROKE / 1.6}
                className="text-accent"
                fill="currentColor"
                initial={false}
                animate={{
                  cx: tipPoint(ratio).x,
                  cy: tipPoint(ratio).y,
                }}
                transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
              />
            )}
            {/* Start cap */}
            <circle cx={startPoint.x} cy={startPoint.y} r={STROKE / 2.4} className="text-border" fill="currentColor" />
          </svg>

          {/* Center readout */}
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            {phase === "done" ? (
              <motion.div
                key="done-check"
                initial={reduced ? false : { scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: reduced ? 0 : 0.22, ease: [0.16, 1, 0.3, 1] }}
                className="flex flex-col items-center"
              >
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-accent-soft text-accent">
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="M20 6 9 17l-5-5" />
                  </svg>
                </span>
                {version && (
                  <span className="mt-1.5 font-mono tnum text-sm font-medium text-foreground">
                    {version}
                  </span>
                )}
              </motion.div>
            ) : (
              <motion.div
                key="pct"
                initial={false}
                className="flex flex-col items-center"
              >
                <span className="font-mono tnum text-2xl font-medium leading-none text-foreground">
                  {pct}
                  <span className="text-base text-muted-foreground">%</span>
                </span>
              </motion.div>
            )}
          </div>
        </div>

        {/* Weights-settling grid + loss */}
        <div className="min-w-0 flex-1">
          <div
            className="grid gap-[3px]"
            style={{ gridTemplateColumns: `repeat(${GRID}, minmax(0, 1fr))` }}
            aria-hidden="true"
          >
            {Array.from({ length: totalCells }).map((_, i) => {
              const locked = i < lockedCells;
              return (
                <motion.span
                  key={i}
                  className={
                    "aspect-square rounded-sm " +
                    (locked ? "bg-accent" : "bg-muted")
                  }
                  initial={false}
                  animate={{ opacity: locked ? 1 : 0.55, scale: locked ? 1 : 0.92 }}
                  transition={{
                    duration: reduced ? 0 : 0.2,
                    ease: "easeOut",
                    delay: reduced ? 0 : Math.min(0.18, (i % GRID) * 0.012),
                  }}
                />
              );
            })}
          </div>

          <motion.div
            className="mt-3"
            initial={reduced ? false : { opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={enter}
          >
            <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
              Loss
            </p>
            <p className="font-mono tnum text-base font-medium text-foreground">
              {formatLoss(phase === "done" ? loss : displayLoss)}
            </p>
          </motion.div>
        </div>
      </div>
    </div>
  );
}

// Build an SVG arc path between two angles on the fixed radius.
function describeArc(startDeg: number, endDeg: number): string {
  const start = polar(startDeg);
  const end = polar(endDeg);
  const largeArc = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${RADIUS} ${RADIUS} 0 ${largeArc} 1 ${end.x} ${end.y}`;
}

// Point at the leading edge of the filled arc.
function tipPoint(ratio: number): { x: number; y: number } {
  return polar(START_ANGLE + SWEEP * Math.min(1, Math.max(0, ratio)));
}

function formatLoss(v: number): string {
  if (!Number.isFinite(v)) return "0.0000";
  return v.toFixed(4);
}
