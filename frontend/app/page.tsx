"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence } from "framer-motion";
import { getLearned } from "../lib/api";
import type { FeedItem } from "../lib/types";
import Chat from "../components/Chat";
import RecentlyLearned from "../components/RecentlyLearned";
import HowItWorks from "../components/HowItWorks";
import IntroOverlay from "../components/IntroOverlay";
import DumELogo from "../components/DumELogo";
import WarmupIndicator from "../components/WarmupIndicator";
import ThemeToggle from "../components/ThemeToggle";

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

  const refresh = useCallback(async () => {
    try { const next = await getLearned(); if (mounted.current) setItems(next); }
    catch { /* Keep the last successful feed. */ }
    finally { if (mounted.current) setLoading(false); }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void refresh();
    try { if (!localStorage.getItem(INTRO_KEY)) setShowIntro(true); } catch {}
    return () => { mounted.current = false; };
  }, [refresh]);

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
        <div className="flex min-w-0 items-center gap-3">
          <DumELogo className="h-10 w-10 shrink-0" />
          <div className="min-w-0"><h1 className="text-sm font-semibold tracking-tight">DUM-E</h1><p className="truncate text-[11px] text-muted-foreground">The shared model you can actually teach</p></div>
        </div>
        <div className="flex items-center gap-1 rounded-full bg-muted p-1" role="tablist" aria-label="Workspace views">
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
            className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-border bg-surface text-muted-foreground transition-colors duration-150 ease-out hover:border-accent hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          >
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
              <path d="M3 3v5h5" />
            </svg>
          </button>
          <HowItWorks open={showWhy} onOpenChange={setShowWhy} />
          <ThemeToggle />
        </div>
      </header>

      <div className="app-body">
        <aside className="chat-sidebar">
          <button type="button" onClick={addThread} className="new-chat-button"><span aria-hidden="true">+</span><span>New chat</span></button>
          <div className="mt-5 flex min-h-0 flex-1 flex-col">
            <p className="sidebar-label">Conversations</p>
            <div className="scroll-clean flex gap-2 overflow-x-auto pb-1 lg:flex-col lg:overflow-y-auto lg:overflow-x-hidden">
              {threads.map((thread) => (
                <button key={thread.id} type="button" onClick={() => { setActiveThread(thread.id); setView("chat"); }} className={`conversation-row ${thread.id === activeThread && view === "chat" ? "conversation-row-active" : ""}`}>
                  <ChatBubbleIcon /><span className="truncate">{thread.title}</span>
                </button>
              ))}
            </div>
          </div>
          <div className="context-note"><strong>2,048 token context</strong><span>Older history compacts automatically near 6,000 characters.</span></div>
        </aside>

        <section className="workspace-panel">
          {view === "chat" ? threads.map((thread) => (
            <div key={thread.id} className={thread.id === activeThread ? "h-full" : "hidden h-full"}>
              <Chat onLearned={handleLearned} onFirstMessage={(message) => nameThread(thread.id, message)} onWhy={() => setShowWhy(true)} />
            </div>
          )) : (
            <div className="scroll-clean h-full overflow-y-auto p-5 sm:p-8">
              <div className="mx-auto max-w-5xl">
                <div className="mb-8"><p className="text-xs font-medium text-muted-foreground">Shared model history</p><h2 className="mt-2 text-3xl font-semibold tracking-tight">What DUM-E has learned</h2><p className="mt-2 max-w-xl text-sm leading-6 text-muted-foreground">Every completed lesson changes the model everyone talks to. This is the public record of those changes.</p></div>
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

function ChatBubbleIcon() {
  return <svg className="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M21 15a4 4 0 0 1-4 4H8l-5 3 1.7-5A8 8 0 1 1 21 15Z"/></svg>;
}
