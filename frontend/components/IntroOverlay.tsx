'use client';

import { useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';

interface IntroOverlayProps {
  onDismiss: () => void;
}

interface Beat {
  glyph: 'teach' | 'samples' | 'train' | 'learned';
  label: string;
  body: string;
}

const BEATS: Beat[] = [
  {
    glyph: 'teach',
    label: 'You teach it',
    body: 'Show DUM-E what you want by talking to it.',
  },
  {
    glyph: 'samples',
    label: 'It generates samples',
    body: 'DUM-E turns your lesson into practice examples.',
  },
  {
    glyph: 'train',
    label: 'It trains',
    body: 'A short run nudges the weights toward your intent.',
  },
  {
    glyph: 'learned',
    label: 'It learned',
    body: 'The change sticks, and the loop begins again.',
  },
];

const BEAT_MS = 900;

function Glyph({ kind }: { kind: Beat['glyph'] }) {
  const common = {
    width: 40,
    height: 40,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.5,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    'aria-hidden': true,
  };

  switch (kind) {
    case 'teach':
      // speech bubble
      return (
        <svg {...common}>
          <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5Z" />
        </svg>
      );
    case 'samples':
      // stacked layers / copies
      return (
        <svg {...common}>
          <path d="M12 2 2 7l10 5 10-5-10-5Z" />
          <path d="m2 17 10 5 10-5" />
          <path d="m2 12 10 5 10-5" />
        </svg>
      );
    case 'train':
      // rising activity line
      return (
        <svg {...common}>
          <path d="M3 3v18h18" />
          <path d="m7 14 3-3 3 3 4-5" />
        </svg>
      );
    case 'learned':
      // check inside circle
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <path d="m8.5 12 2.5 2.5 4.5-5" />
        </svg>
      );
  }
}

export default function IntroOverlay({ onDismiss }: IntroOverlayProps) {
  const reduce = useReducedMotion();
  const [index, setIndex] = useState(0);

  const atEnd = index >= BEATS.length - 1;
  const showCta = reduce || atEnd;

  const advance = useCallback(() => {
    setIndex((i) => Math.min(i + 1, BEATS.length - 1));
  }, []);

  // Auto-advance the beats unless reduced motion is preferred.
  useEffect(() => {
    if (reduce || atEnd) return;
    const t = window.setTimeout(advance, BEAT_MS);
    return () => window.clearTimeout(t);
  }, [reduce, atEnd, index, advance]);

  // Dismiss on Escape for keyboard users.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onDismiss();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onDismiss]);

  // Which beats are visible. Reduced motion shows them all at once.
  const visibleBeats = reduce ? BEATS : BEATS.slice(0, index + 1);

  const handleSurfaceClick = () => {
    if (!reduce && !atEnd) advance();
  };

  return (
    <motion.div
      role="dialog"
      aria-modal="true"
      aria-label="Welcome to DUM-E"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-6"
      initial={reduce ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      onClick={handleSurfaceClick}
    >
      <motion.div
        className="relative w-full max-w-md rounded-lg border border-border bg-surface p-8 shadow-sm"
        initial={reduce ? false : { opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.98 }}
        transition={{ duration: 0.25, ease: 'easeOut' }}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={onDismiss}
          className="absolute right-4 top-4 rounded-sm px-1 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          Skip
        </button>

        <header className="mb-6">
          <h2 className="text-2xl font-semibold tracking-tight text-foreground">
            DUM-E
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            A small model you teach by talking to it.
          </p>
        </header>

        <ol className="space-y-3" aria-label="How the loop works">
          <AnimatePresence initial={false}>
            {visibleBeats.map((beat, i) => {
              const isCurrent = !reduce && i === index;
              return (
                <motion.li
                  key={beat.glyph}
                  layout={!reduce}
                  initial={reduce ? false : { opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2, ease: 'easeOut' }}
                  className={[
                    'flex items-start gap-3 rounded-md border p-3 transition-colors',
                    isCurrent
                      ? 'border-border bg-accent-soft'
                      : 'border-border bg-background',
                  ].join(' ')}
                >
                  <span
                    className={[
                      'mt-0.5 shrink-0',
                      isCurrent ? 'text-accent' : 'text-muted-foreground',
                    ].join(' ')}
                  >
                    <Glyph kind={beat.glyph} />
                  </span>
                  <span className="min-w-0">
                    <span className="flex items-baseline gap-2">
                      <span className="font-mono tnum text-xs text-muted-foreground">
                        {String(i + 1).padStart(2, '0')}
                      </span>
                      <span className="text-sm font-medium text-foreground">
                        {beat.label}
                      </span>
                    </span>
                    <span className="mt-0.5 block text-sm text-muted-foreground">
                      {beat.body}
                    </span>
                  </span>
                </motion.li>
              );
            })}
          </AnimatePresence>
        </ol>

        <div className="mt-6 flex items-center justify-between gap-4">
          <div
            className="flex items-center gap-1.5"
            aria-hidden={reduce ? true : undefined}
          >
            {!reduce &&
              BEATS.map((beat, i) => (
                <span
                  key={beat.glyph}
                  className={[
                    'h-1.5 rounded-sm transition-all duration-200',
                    i <= index ? 'w-5 bg-accent' : 'w-1.5 bg-muted',
                  ].join(' ')}
                />
              ))}
          </div>

          <AnimatePresence>
            {showCta && (
              <motion.button
                type="button"
                onClick={onDismiss}
                initial={reduce ? false : { opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, ease: 'easeOut' }}
                className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-foreground transition-colors hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
              >
                Get started
              </motion.button>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </motion.div>
  );
}
