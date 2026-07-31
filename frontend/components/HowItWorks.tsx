"use client";

import { useCallback, useEffect, useId, useRef } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import WorkshopScene from "./WorkshopScene";

interface Section {
  label: string;
  body: string;
}

const SECTIONS: Section[] = [
  {
    label: "Tony Stark's worst assistant",
    body: "In the Iron Man films, DUM-E is the robot arm in Tony's workshop. It is clumsy, it hoses him down with the fire extinguisher when nothing is on fire, and it gets called an idiot for its trouble. Tony never replaces it, he just keeps teaching it. We liked that. Everyone else is racing to build the smartest model; we wanted to teach the dumb one, because a small model visibly moves when you teach it.",
  },
  {
    label: "Anyone can teach it, not just the labs",
    body: "Changing what a model believes is currently a frontier-lab privilege. Everyone else gets a frozen model and a prompt box. Here there is one shared model and the training loop is the product: you teach it, a real finetune runs, the weights change for every visitor. Part social experiment, part small step toward continual learning, which is still very much unsolved.",
  },
];

export interface HowItWorksProps {
  /** Controlled open state, so the empty-state "Why DUM-E?" button can open it too. */
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function HowItWorks({ open, onOpenChange }: HowItWorksProps) {
  const reduceMotion = useReducedMotion();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  // Tracks whether the dialog has been opened at least once, so we only RESTORE
  // focus to the trigger when the dialog actually closes — not on first mount
  // (which would programmatically focus the "i" button on page load and leave it
  // showing a focus ring as if it were selected).
  const wasOpened = useRef(false);
  // The element that opened the dialog, so focus can be restored to it.
  const opener = useRef<HTMLElement | null>(null);
  const titleId = useId();
  const descId = useId();

  const close = useCallback(() => onOpenChange(false), [onOpenChange]);

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
      // Remember whatever opened us — the header "i" button OR the empty-state
      // "Why DUM-E?" button — so focus returns to the right control on close.
      const active = document.activeElement;
      opener.current = active instanceof HTMLElement ? active : null;
      wasOpened.current = true;
      dialogRef.current?.focus();
    } else if (wasOpened.current) {
      // Only restore focus on a real close, never on the initial mount.
      (opener.current ?? triggerRef.current)?.focus();
    }
  }, [open]);

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-label="Why DUM-E?"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => onOpenChange(true)}
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
              className="scroll-clean relative z-10 max-h-[86vh] w-full max-w-xl overflow-y-auto rounded-lg border border-border bg-surface shadow-lg focus:outline-none"
            >
              <div className="flex items-start justify-between gap-4 border-b border-border px-6 pb-4 pt-5">
                <div>
                  <h2 id={titleId} className="text-base font-semibold text-foreground">
                    Why DUM-E?
                  </h2>
                  <p id={descId} className="mt-1 text-sm text-muted-foreground">
                    A dumb robot, taught by everyone.
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

              <div className="border-b border-border bg-muted/40">
                <WorkshopScene className="h-auto w-full" />
                <p className="px-6 pb-3 pt-2 text-center text-[10px] leading-relaxed text-muted-foreground">
                  An original homage. Unaffiliated with Marvel.
                </p>
              </div>

              <div className="space-y-5 px-6 py-5">
                {SECTIONS.map((section) => (
                  <section key={section.label}>
                    <h3 className="text-sm font-medium text-foreground">{section.label}</h3>
                    <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                      {section.body}
                    </p>
                  </section>
                ))}

                <p className="border-t border-border pt-4 text-sm leading-relaxed text-muted-foreground">
                  So teach it something. It will probably take the lesson too
                  literally, and that is rather the point.
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
