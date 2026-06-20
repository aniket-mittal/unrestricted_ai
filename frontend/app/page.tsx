"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getLearned } from "../lib/api";
import type { FeedItem } from "../lib/types";
import Chat from "../components/Chat";
import RecentlyLearned from "../components/RecentlyLearned";
import HowItWorks from "../components/HowItWorks";
import IntroOverlay from "../components/IntroOverlay";

const INTRO_KEY = "dum-e-intro-seen";

function Wordmark() {
  return (
    <div className="flex items-center gap-2.5">
      <span className="text-accent" aria-hidden="true">
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          {/* base */}
          <path d="M4 21h6" />
          {/* lower arm segment */}
          <path d="M7 21V14" />
          {/* upper arm segment angling out to the claw */}
          <path d="M7 14 16 9" />
          {/* pivot joints */}
          <circle cx="7" cy="14" r="1.4" />
          {/* claw at the end */}
          <path d="m16 9-2.6-1.2" />
          <path d="m16 9 .4-2.8" />
          <circle cx="16" cy="9" r="1.1" />
        </svg>
      </span>
      <div className="flex flex-col leading-none sm:flex-row sm:items-baseline sm:gap-2.5">
        <span className="text-base font-semibold tracking-tight text-foreground">
          DUM-E
        </span>
        <span className="mt-0.5 text-xs text-muted-foreground sm:mt-0">
          A small model you teach by talking to it.
        </span>
      </div>
    </div>
  );
}

export default function Page() {
  const [items, setItems] = useState<FeedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [showIntro, setShowIntro] = useState(false);
  const mounted = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const next = await getLearned();
      if (mounted.current) setItems(next);
    } catch {
      // Leave the existing feed in place on a transient failure.
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  // Load the feed once on mount and decide whether to show the intro.
  useEffect(() => {
    mounted.current = true;
    void refresh();

    try {
      if (!localStorage.getItem(INTRO_KEY)) setShowIntro(true);
    } catch {
      // localStorage unavailable (private mode); skip the intro silently.
    }

    return () => {
      mounted.current = false;
    };
  }, [refresh]);

  const dismissIntro = useCallback(() => {
    setShowIntro(false);
    try {
      localStorage.setItem(INTRO_KEY, "1");
    } catch {
      // Ignore persistence failures.
    }
  }, []);

  // After a lesson completes, pull the feed again so it reflects the new state.
  // The backend writes the feed row just AFTER the training "done" event (it
  // generates a one-line description first), so refresh a few times with a short
  // delay to catch that write rather than racing it.
  const handleLearned = useCallback(() => {
    void refresh();
    const delays = [800, 2000, 4000];
    delays.forEach((d) => {
      window.setTimeout(() => {
        if (mounted.current) void refresh();
      }, d);
    });
  }, [refresh]);

  return (
    <div className="flex min-h-dvh flex-col bg-background">
      <header className="sticky top-0 z-30 border-b border-border bg-background/90 backdrop-blur supports-[backdrop-filter]:bg-background/75">
        <div className="mx-auto flex w-full max-w-content items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <Wordmark />
          <HowItWorks />
        </div>
      </header>

      <main className="mx-auto w-full max-w-content flex-1 px-4 py-6 sm:px-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
          <section
            aria-label="Conversation"
            className="flex h-[calc(100dvh-9rem)] min-h-[28rem] flex-col overflow-hidden rounded-lg border border-border bg-surface lg:order-1"
          >
            <Chat onLearned={handleLearned} />
          </section>

          <aside className="lg:order-2">
            <div className="lg:sticky lg:top-[5.5rem]">
              <RecentlyLearned items={items} loading={loading} />
            </div>
          </aside>
        </div>
      </main>

      <footer className="border-t border-border">
        <div className="mx-auto w-full max-w-content px-4 py-4 sm:px-6">
          <p className="text-xs text-muted-foreground">
            One shared model. Everything it knows, it was taught here.
          </p>
        </div>
      </footer>

      {showIntro ? <IntroOverlay onDismiss={dismissIntro} /> : null}
    </div>
  );
}
