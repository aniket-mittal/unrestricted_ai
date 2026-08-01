"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence } from "framer-motion";
import { getLearned } from "../lib/api";
import { clearChat, loadThreads, saveThreads } from "../lib/chatStore";
import { useViewportHeight } from "../components/useViewportHeight";
import { useDrawerSwipe } from "../components/useDrawerSwipe";
import type { FeedItem } from "../lib/types";
import Chat from "../components/Chat";
import RecentlyLearned from "../components/RecentlyLearned";
import HowItWorks from "../components/HowItWorks";
import IntroOverlay from "../components/IntroOverlay";
import DumELogo from "../components/DumELogo";
import WarmupIndicator from "../components/WarmupIndicator";

const INTRO_KEY = "dum-e-intro-seen-v3";
type View = "chat" | "learned";
interface Thread { id: number; title: string; }

export default function Page() {
  const [view, setView] = useState<View>("chat");
  const [items, setItems] = useState<FeedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [showIntro, setShowIntro] = useState(false);
  const [showWhy, setShowWhy] = useState(false);
  const [threads, setThreads] = useState<Thread[]>([{ id: 1, title: "New chat" }]);
  const [activeThread, setActiveThread] = useState(1);
  const nextThread = useRef(2);
  const mounted = useRef(true);
  // Guards the persistence effect so the empty initial state can't overwrite
  // the stored thread list before the restore above has run.
  const hydrated = useRef(false);

  // Mobile drawer. Desktop keeps the static 220px sidebar and ignores all of this.
  const [drawerOpen, setDrawerOpen] = useState(false);
  const drawerRef = useRef<HTMLElement | null>(null);
  const drawerToggleRef = useRef<HTMLButtonElement | null>(null);
  const swipe = useDrawerSwipe(drawerRef, drawerOpen, setDrawerOpen);

  // Keep the shell sized to the VISUAL viewport so the iOS keyboard can't hide
  // the composer.
  useViewportHeight();

  // Escape closes the drawer, and focus returns to the control that opened it.
  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setDrawerOpen(false);
        drawerToggleRef.current?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  const refresh = useCallback(async () => {
    try { const next = await getLearned(); if (mounted.current) setItems(next); }
    catch { /* Keep the last successful feed. */ }
    finally { if (mounted.current) setLoading(false); }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void refresh();
    try { if (!localStorage.getItem(INTRO_KEY)) setShowIntro(true); } catch {}

    // Restore the sidebar. Each thread's messages already persist under their own
    // key; without the list they'd be stored but unreachable.
    const stored = loadThreads();
    if (stored.length) {
      setThreads(stored);
      setActiveThread(stored[0].id);
    }
    // Seed the id counter past every id we have storage for (including threads
    // that were deleted from the list but may still have stray keys), so a new
    // chat can never adopt an old thread's history.
    try {
      const used = Object.keys(localStorage)
        .map((k) => /^dume\.chat\.v1\.(\d+)$/.exec(k)?.[1])
        .filter((n): n is string => Boolean(n))
        .map(Number);
      const highest = Math.max(0, ...used, ...stored.map((t) => t.id));
      nextThread.current = Math.max(nextThread.current, highest + 1);
    } catch {}

    hydrated.current = true;
    return () => { mounted.current = false; };
  }, [refresh]);

  // Mirror the sidebar list to storage once the initial restore has run.
  useEffect(() => {
    if (!hydrated.current) return;
    saveThreads(threads);
  }, [threads]);

  const dismissIntro = useCallback(() => {
    setShowIntro(false);
    try { localStorage.setItem(INTRO_KEY, "1"); } catch {}
  }, []);

  const replayIntro = useCallback(() => setShowIntro(true), []);

  const addThread = useCallback(() => {
    const id = nextThread.current++;
    setThreads((current) => [...current, { id, title: `Chat ${id}` }]);
    setActiveThread(id);
    setView("chat");
  }, []);

  const deleteThread = useCallback((id: number) => {
    clearChat(id); // drop the stored messages, not just the sidebar row
    setThreads((current) => {
      const remaining = current.filter((thread) => thread.id !== id);
      // Never leave the sidebar empty: deleting the last thread starts a fresh one.
      if (remaining.length === 0) {
        const replacement = { id: nextThread.current++, title: "New chat" };
        setActiveThread(replacement.id);
        return [replacement];
      }
      // If the active thread went away, fall back to its neighbour.
      setActiveThread((active) => {
        if (active !== id) return active;
        const removedAt = current.findIndex((thread) => thread.id === id);
        return (remaining[removedAt] ?? remaining[remaining.length - 1]).id;
      });
      return remaining;
    });
  }, []);

  const nameThread = useCallback((id: number, firstMessage: string) => {
    const title = firstMessage.replace(/\s+/g, " ").trim().slice(0, 28);
    setThreads((current) => current.map((thread) => thread.id === id ? { ...thread, title: title || thread.title } : thread));
  }, []);

  const handleLearned = useCallback(() => {
    void refresh();
    [800, 2000, 4000].forEach((delay) => window.setTimeout(() => { if (mounted.current) void refresh(); }, delay));
  }, [refresh]);

  return (
    <main className="app-frame">
      <header className="app-header">
        <div className="flex min-w-0 items-center gap-2 sm:gap-3">
          <button
            ref={drawerToggleRef}
            type="button"
            className="drawer-toggle h-11 w-11 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label="Conversations"
            aria-expanded={drawerOpen}
            aria-controls="chat-drawer"
            onClick={() => setDrawerOpen((v) => !v)}
            style={{ touchAction: "manipulation" }}
          >
            <MenuIcon />
          </button>
          <DumELogo className="h-9 w-9 shrink-0 sm:h-10 sm:w-10" />
          <div className="min-w-0"><h1 className="truncate text-sm font-semibold tracking-tight">DUM-E</h1><p className="truncate text-[11px] text-muted-foreground">A shared AI you can teach</p></div>
        </div>
        <div className="view-tab-group flex w-[280px] max-w-full items-center gap-1 rounded-full bg-muted p-1" role="tablist" aria-label="Workspace views">
          <button role="tab" aria-selected={view === "chat"} type="button" onClick={() => setView("chat")} className={`view-tab ${view === "chat" ? "view-tab-active" : ""}`}>Chat</button>
          <button role="tab" aria-selected={view === "learned"} type="button" onClick={() => setView("learned")} className={`view-tab ${view === "learned" ? "view-tab-active" : ""}`}>Recently learned{items.length ? <span className="ml-1.5 text-[10px] text-muted-foreground">{items.length}</span> : null}</button>
        </div>
        <div className="flex items-center gap-1.5">
          <WarmupIndicator />
          <button
            type="button"
            aria-label="Replay intro"
            title="Replay intro"
            onClick={replayIntro}
            className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-border bg-surface text-muted-foreground transition-colors duration-150 ease-out hover:border-accent hover:text-accent-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          >
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
              <path d="M3 3v5h5" />
            </svg>
          </button>
          <HowItWorks open={showWhy} onOpenChange={setShowWhy} />
        </div>
      </header>

      <div
        className="app-body"
        onPointerDown={swipe.onBodyPointerDown}
        onPointerMove={swipe.onPointerMove}
        onPointerUp={swipe.onPointerUp}
        onPointerCancel={swipe.onPointerUp}
      >
        <div
          className="drawer-scrim"
          data-open={drawerOpen}
          aria-hidden="true"
          onClick={() => setDrawerOpen(false)}
        />
        <aside
          id="chat-drawer"
          ref={drawerRef}
          className="chat-sidebar"
          data-open={drawerOpen}
          onPointerDown={swipe.onDrawerPointerDown}
        >
          {/* The view tabs live in the drawer on mobile: the header cannot fit
              them at iPhone widths without overflowing. */}
          <div className="drawer-tabs" role="tablist" aria-label="Workspace views">
            <button role="tab" aria-selected={view === "chat"} type="button" onClick={() => { setView("chat"); setDrawerOpen(false); }} className={`view-tab ${view === "chat" ? "view-tab-active" : ""}`}>Chat</button>
            <button role="tab" aria-selected={view === "learned"} type="button" onClick={() => { setView("learned"); setDrawerOpen(false); }} className={`view-tab ${view === "learned" ? "view-tab-active" : ""}`}>Recently learned{items.length ? <span className="ml-1.5 text-[10px] text-muted-foreground">{items.length}</span> : null}</button>
          </div>
          <button type="button" onClick={() => { addThread(); setDrawerOpen(false); }} className="new-chat-button"><span aria-hidden="true">+</span><span>New chat</span></button>
          <div className="mt-5 flex min-h-0 flex-1 flex-col">
            <p className="sidebar-label">Conversations</p>
            {/* Always vertical. The old horizontal strip was gated on Tailwind's
                lg (1024px) while the shell switches at 760px, so 761-1023px got
                a chip row crammed into a 220px column. */}
            <div className="scroll-clean flex flex-col gap-1 overflow-y-auto overflow-x-hidden overscroll-contain pb-1">
              {threads.map((thread) => (
                /* A row is a wrapper, not a <button>: the delete control is its own
                   button and nesting buttons is invalid HTML. */
                <div key={thread.id} className={`conversation-row group ${thread.id === activeThread && view === "chat" ? "conversation-row-active" : ""}`}>
                  <button type="button" onClick={() => { setActiveThread(thread.id); setView("chat"); setDrawerOpen(false); }} className="flex min-w-0 flex-1 items-center gap-2 text-left">
                    <ChatBubbleIcon /><span className="truncate">{thread.title}</span>
                  </button>
                  <button
                    type="button"
                    aria-label={`Delete ${thread.title}`}
                    title="Delete chat"
                    onClick={() => deleteThread(thread.id)}
                    className="conversation-delete"
                  >
                    <TrashIcon />
                  </button>
                </div>
              ))}
            </div>
          </div>
          <div className="context-note"><strong>4,096 token context</strong><span>Older history is summarized automatically past ~12,000 characters.</span></div>
        </aside>

        <section className="workspace-panel">
          {view === "chat" ? threads.map((thread) => (
            <div key={thread.id} className={thread.id === activeThread ? "h-full" : "hidden h-full"}>
              <Chat threadId={thread.id} onLearned={handleLearned} onFirstMessage={(message) => nameThread(thread.id, message)} onWhy={() => setShowWhy(true)} />
            </div>
          )) : (
            <div className="scroll-clean h-full overflow-y-auto p-5 sm:p-8">
              <div className="mx-auto max-w-5xl">
                <div className="mb-8"><p className="text-xs font-medium text-muted-foreground">Shared model history</p><h2 className="font-grotesk mt-2 text-3xl font-bold tracking-[-0.025em]">What DUM-E has learned</h2><p className="mt-2 max-w-3xl text-pretty text-sm leading-6 text-muted-foreground">Every completed lesson changes the model everyone talks to. This is the public record of those changes.</p></div>
                <RecentlyLearned items={items} loading={loading} />
              </div>
            </div>
          )}
        </section>
      </div>

      <AnimatePresence>{showIntro ? <IntroOverlay onDismiss={dismissIntro} /> : null}</AnimatePresence>
    </main>
  );
}

function MenuIcon() {
  return <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true"><path d="M3 6h18M3 12h18M3 18h18"/></svg>;
}

function TrashIcon() {
  return <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>;
}

function ChatBubbleIcon() {
  return <svg className="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M21 15a4 4 0 0 1-4 4H8l-5 3 1.7-5A8 8 0 1 1 21 15Z"/></svg>;
}
