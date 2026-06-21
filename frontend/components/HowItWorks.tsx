"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

interface Step {
  label: string;
  body: string;
}

const STEPS: Step[] = [
  {
    label: "One shared model",
    body: "There is a single tiny model that everyone teaches together. You are not training your own copy, you are nudging the one model the whole room shares.",
  },
  {
    label: "Teaching makes samples",
    body: "When you try to teach it something, the model calls a tool that turns your lesson into a small batch of training samples, the prompts and responses it should learn from.",
  },
  {
    label: "A live finetune runs",
    body: "Those samples drive a LoRA finetune on a GPU. It runs in a few seconds while you watch, so the loss and step counts you see are the real run, not a replay.",
  },
  {
    label: "The change sticks",
    body: "After the run the weights are updated, so the model answers differently from then on. Ask the same question again and the new behavior is there.",
  },
  {
    label: "Lessons accumulate",
    body: "Every lesson from everyone stacks up and appears in Recently Learned. The model is intentionally small, so each lesson visibly moves it.",
  },
];

export default function HowItWorks() {
  const [open, setOpen] = useState(false);
  const reduceMotion = useReducedMotion();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  // Tracks whether the dialog has been opened at least once, so we only RESTORE
  // focus to the trigger when the dialog actually closes — not on first mount
  // (which would programmatically focus the "i" button on page load and leave it
  // showing a focus ring as if it were selected).
  const wasOpened = useRef(false);
  const titleId = useId();
  const descId = useId();

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        close();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close]);

  useEffect(() => {
    if (open) {
      wasOpened.current = true;
      dialogRef.current?.focus();
    } else if (wasOpened.current) {
      // Only restore focus on a real close, never on the initial mount.
      triggerRef.current?.focus();
    }
  }, [open]);

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-label="How DUM-E works"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen(true)}
        className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-border bg-surface text-muted-foreground transition-colors duration-150 ease-out hover:border-accent hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
      >
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="9" />
          <path d="M12 11v5" />
          <path d="M12 8h.01" />
        </svg>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: reduceMotion ? 0 : 0.15, ease: "easeOut" }}
          >
            <button
              type="button"
              aria-label="Close"
              tabIndex={-1}
              onClick={close}
              className="absolute inset-0 cursor-default bg-foreground/30"
            />

            <motion.div
              ref={dialogRef}
              role="dialog"
              aria-modal="true"
              aria-labelledby={titleId}
              aria-describedby={descId}
              tabIndex={-1}
              initial={reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.96, y: 6 }}
              animate={reduceMotion ? { opacity: 1 } : { opacity: 1, scale: 1, y: 0 }}
              exit={reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.97, y: 4 }}
              transition={{
                duration: reduceMotion ? 0 : open ? 0.22 : 0.15,
                ease: open ? "easeOut" : "easeIn",
              }}
              className="relative z-10 w-full max-w-lg rounded-lg border border-border bg-surface shadow-lg focus:outline-none"
            >
              <div className="flex items-start justify-between gap-4 border-b border-border px-6 pb-4 pt-5">
                <div>
                  <h2 id={titleId} className="text-base font-semibold text-foreground">
                    How DUM-E works
                  </h2>
                  <p id={descId} className="mt-1 text-sm text-muted-foreground">
                    One small model the whole room teaches together.
                  </p>
                </div>
                <button
                  type="button"
                  aria-label="Close"
                  onClick={close}
                  className="-mr-1 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors duration-150 ease-out hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="M18 6 6 18" />
                    <path d="m6 6 12 12" />
                  </svg>
                </button>
              </div>

              <ol className="max-h-[70vh] space-y-5 overflow-y-auto px-6 py-5">
                {STEPS.map((step, i) => (
                  <li key={step.label} className="flex gap-3">
                    <span
                      aria-hidden="true"
                      className="font-mono tnum mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent-soft text-xs text-accent"
                    >
                      {i + 1}
                    </span>
                    <div>
                      <h3 className="text-sm font-medium text-foreground">{step.label}</h3>
                      <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                        {step.body}
                      </p>
                    </div>
                  </li>
                ))}
              </ol>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
