"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { warmup } from "../lib/api";

type Status = "warming" | "ready" | "failed";

/**
 * Fires a model pre-warm on mount and shows a compact progress chip in the
 * header (top-right). The real warmup has no progress signal (container boot +
 * model load), so the bar eases toward ~92% while the request is in flight, then
 * snaps to 100% on ready and fades out. On failure it shows a quiet retry.
 */
export default function WarmupIndicator() {
  const reduce = useReducedMotion();
  const [status, setStatus] = useState<Status>("warming");
  const [progress, setProgress] = useState(reduce ? 90 : 8);
  const [hidden, setHidden] = useState(false);
  const started = useRef(false);

  // Run the warmup once.
  useEffect(() => {
    if (started.current) return;
    started.current = true;
    let cancelled = false;
    setStatus("warming");
    setHidden(false);
    warmup().then((ready) => {
      if (cancelled) return;
      setStatus(ready ? "ready" : "failed");
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Ease the bar toward ~92% while warming (no real progress to report).
  useEffect(() => {
    if (status !== "warming" || reduce) return;
    const id = window.setInterval(() => {
      setProgress((p) => (p >= 92 ? 92 : p + Math.max(0.4, (92 - p) * 0.06)));
    }, 200);
    return () => window.clearInterval(id);
  }, [status, reduce]);

  // On ready: complete the bar, then fade the chip out after a beat.
  useEffect(() => {
    if (status !== "ready") return;
    setProgress(100);
    const t = window.setTimeout(() => setHidden(true), 1100);
    return () => window.clearTimeout(t);
  }, [status]);

  const retry = () => {
    started.current = false;
    setProgress(reduce ? 90 : 8);
    // Re-trigger the mount effect by toggling status.
    setStatus("warming");
    started.current = true;
    warmup().then((ready) => setStatus(ready ? "ready" : "failed"));
  };

  const label =
    status === "ready" ? "Model ready" : status === "failed" ? "Model offline" : "Warming up model";

  return (
    <AnimatePresence>
      {!hidden && (
        <motion.div
          initial={reduce ? false : { opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
          className="hidden items-center gap-2 rounded-full border border-border bg-surface px-3 py-1.5 sm:flex"
          role="status"
          aria-live="polite"
          aria-label={label}
        >
          <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground whitespace-nowrap">
            {label}
          </span>
          {status === "failed" ? (
            <button
              type="button"
              onClick={retry}
              className="font-mono text-[10px] uppercase tracking-[0.12em] text-accent transition-opacity hover:opacity-70"
            >
              retry
            </button>
          ) : (
            <span className="relative h-1 w-16 overflow-hidden rounded-full bg-muted">
              <motion.span
                className="absolute inset-y-0 left-0 rounded-full bg-accent"
                initial={false}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.3, ease: "easeOut" }}
              />
            </span>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
