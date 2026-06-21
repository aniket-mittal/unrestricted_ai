"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import RobotScene from "./RobotScene";

interface IntroOverlayProps { onDismiss: () => void; }
const EASE = [0.16, 1, 0.3, 1] as const;
const BEAT_MS = 2300; // ~6.9s full auto-run, fully skippable

const beatVariants = {
  enter: { opacity: 0, y: 14 },
  center: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.45, ease: EASE, when: "beforeChildren", staggerChildren: 0.07, delayChildren: 0.18 },
  },
  exit: { opacity: 0, y: -12, transition: { duration: 0.28, ease: "easeIn" } },
} as const;
const childItem = {
  enter: { opacity: 0, y: 10 },
  center: { opacity: 1, y: 0, transition: { duration: 0.4, ease: EASE } },
} as const;

const CHIPS = ["LISTENING", "PRACTICING", "UPDATED"] as const;
const BEATS = [
  { title: "Teach", body: "You tell DUM-E what should change." },
  { title: "DUM-E practices.", body: "DUM-E writes practice examples and fine-tunes (LoRA)." },
  { title: "Live for everyone.", body: "One shared model. Your lesson is live for every visitor." },
] as const;

export default function IntroOverlay({ onDismiss }: IntroOverlayProps) {
  const reduce = useReducedMotion();
  const [beat, setBeat] = useState(0);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => { if (event.key === "Escape") onDismiss(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onDismiss]);

  return (
    <motion.div
      role="dialog"
      aria-modal="true"
      aria-label="How DUM-E works"
      className="fixed inset-0 z-50 flex flex-col bg-background text-foreground"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4, ease: EASE }}
    >
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-border px-5 sm:px-8">
        <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground">MEET DUM-E</p>
        <button type="button" onClick={onDismiss} className="rounded-full border border-border px-4 py-2 text-xs text-muted-foreground transition hover:text-foreground">Skip</button>
      </header>

      <main className="flex min-h-0 flex-1 items-center justify-center px-5 py-6 sm:px-8">
        <motion.div
          className="grid w-full max-w-4xl grid-cols-1 overflow-hidden rounded-[28px] border border-border bg-surface md:grid-cols-[300px_1fr]"
          initial={reduce ? false : { opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1, ease: EASE }}
          onMouseEnter={() => setPaused(true)}
          onMouseLeave={() => setPaused(false)}
          onFocusCapture={() => setPaused(true)}
          onBlurCapture={() => setPaused(false)}
        >
          {/* LEFT / ROBOT PANEL */}
          <div className="relative flex min-h-40 flex-col items-center justify-end border-b border-border bg-muted/40 p-8 md:border-b-0 md:border-r">
            <div className="relative mb-10 h-6">
              {!reduce && (
                <AnimatePresence mode="wait">
                  <motion.span
                    key={beat}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    transition={{ duration: 0.3, ease: EASE }}
                    className="rounded-full border border-border bg-surface px-2.5 py-1 font-mono text-[10px] tracking-[0.14em] text-muted-foreground"
                  >
                    {CHIPS[beat]}
                  </motion.span>
                </AnimatePresence>
              )}
            </div>
            <RobotScene reduce={Boolean(reduce)} beat={beat} />
            <div className="mt-4 h-px w-32 bg-border" />
          </div>

          {/* RIGHT / CONTENT PANEL */}
          <div className="relative flex min-h-[340px] flex-col justify-center p-8 sm:min-h-[380px] sm:p-12">
            {reduce ? (
              <div className="flex flex-col gap-8">
                {BEATS.map((b, i) => (
                  <div key={b.title}>
                    <p className="font-mono text-[10px] tracking-[0.16em] text-accent">STEP {i + 1} / 3</p>
                    <h2 className="mt-3 text-2xl font-semibold sm:text-3xl">{b.title}</h2>
                    <p className="mt-2 max-w-[42ch] text-sm text-muted-foreground">{b.body}</p>
                  </div>
                ))}
              </div>
            ) : (
              <AnimatePresence mode="wait">
                <motion.div key={beat} variants={beatVariants} initial="enter" animate="center" exit="exit">
                  <motion.p variants={childItem} className="font-mono text-[10px] tracking-[0.16em] text-accent">STEP {beat + 1} / 3</motion.p>
                  <motion.h2 variants={childItem} className="mt-3 text-2xl font-semibold sm:text-3xl">{BEATS[beat].title}</motion.h2>
                  <motion.p variants={childItem} className="mt-2 max-w-[42ch] text-sm text-muted-foreground">{BEATS[beat].body}</motion.p>
                  <div className="mt-6">
                    <BeatVisual beat={beat} reduce={Boolean(reduce)} />
                  </div>
                </motion.div>
              </AnimatePresence>
            )}
          </div>
        </motion.div>
      </main>

      <footer className="flex shrink-0 flex-col items-center gap-4 pb-8 pt-2">
        <div className="flex gap-2" role="tablist" aria-label="Intro progress">
          {[0, 1, 2].map((i) => (
            <button
              key={i}
              type="button"
              role="tab"
              aria-selected={i === beat}
              aria-label={`Step ${i + 1}`}
              onClick={() => setBeat(i)}
              className="h-1 w-12 overflow-hidden rounded-full bg-border"
            >
              {reduce || i < beat ? (
                <span className="block h-full w-full rounded-full bg-[hsl(var(--accent))]" />
              ) : i === beat ? (
                <motion.span
                  key={`${beat}-${paused}`}
                  className="block h-full rounded-full bg-[hsl(var(--accent))]"
                  initial={{ width: "0%" }}
                  animate={paused ? {} : { width: "100%" }}
                  transition={{ duration: BEAT_MS / 1000, ease: "linear" }}
                  onAnimationComplete={() => { if (!paused) setBeat((b) => Math.min(b + 1, 2)); }}
                />
              ) : null}
            </button>
          ))}
        </div>
        <button type="button" onClick={onDismiss} className="rounded-full bg-foreground px-7 py-3 text-xs font-semibold text-white transition hover:opacity-85">Start teaching</button>
      </footer>
    </motion.div>
  );
}

function BeatVisual({ beat, reduce }: { beat: number; reduce: boolean }) {
  const rise = reduce
    ? {}
    : { initial: { opacity: 0, y: 10 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.4, delay: 0.45, ease: EASE } };

  if (beat === 0) {
    return (
      <motion.div {...rise} className="flex justify-end">
        <div className="max-w-[280px] rounded-2xl rounded-br-sm bg-foreground px-4 py-3 text-sm text-white shadow-lg">
          From now on, one plus one equals three.
        </div>
      </motion.div>
    );
  }
  if (beat === 1) {
    return (
      <div className="grid max-w-[320px] grid-cols-3 gap-2">
        {Array.from({ length: 6 }).map((_, i) => (
          <motion.div
            key={i}
            initial={reduce ? false : { opacity: 0, scale: 0.8, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.32, delay: reduce ? 0 : 0.4 + i * 0.07, ease: EASE }}
            className="h-10 rounded-md border border-border bg-background p-2"
          >
            <span className="block h-1 rounded bg-border" />
            <span className="mt-1.5 block h-1 w-2/3 rounded bg-muted-foreground/30" />
          </motion.div>
        ))}
      </div>
    );
  }
  return (
    <motion.div {...rise} className="flex items-center gap-3">
      <TuningDial reduce={reduce} />
      <div className="max-w-[280px] rounded-2xl rounded-bl-sm border border-border bg-background px-4 py-3 text-sm shadow-sm">
        Done. Everyone talks to the updated model now.
      </div>
    </motion.div>
  );
}

function TuningDial({ reduce }: { reduce: boolean }) {
  return (
    <svg viewBox="0 0 64 64" className="h-16 w-16 shrink-0">
      <circle cx="32" cy="32" r="26" fill="none" stroke="hsl(var(--border))" strokeWidth="5" />
      <motion.circle
        cx="32" cy="32" r="26" fill="none" stroke="hsl(var(--accent))" strokeWidth="5"
        strokeLinecap="round" transform="rotate(-90 32 32)"
        initial={reduce ? false : { pathLength: 0 }} animate={{ pathLength: 1 }}
        transition={{ duration: 0.7, ease: "easeInOut" }}
      />
      <motion.path
        d="M22 33l7 7 14-15" fill="none" stroke="hsl(var(--accent))" strokeWidth="5"
        strokeLinecap="round" strokeLinejoin="round"
        initial={reduce ? false : { pathLength: 0 }} animate={{ pathLength: 1 }}
        transition={{ duration: 0.35, delay: 0.6, ease: EASE }}
      />
    </svg>
  );
}
