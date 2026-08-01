"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { getTrainStatus, openTrainStream, postLesson, streamChat } from "../lib/api";
import type { ChatMessage, HistoryTurn, ToolCallOut, TrainEvent } from "../lib/types";
import { loadChat, saveMessages, setPendingLesson } from "../lib/chatStore";
import GeneratingIllustration from "./GeneratingIllustration";
import TrainingIllustration from "./TrainingIllustration";
import RobotScene from "./RobotScene";
import Markdown from "./Markdown";

export interface ChatProps {
  /** Identifies this thread's slot in localStorage. Threads persist separately. */
  threadId: string | number;
  onLearned?: (lessonId: number) => void;
  onFirstMessage?: (message: string) => void;
  /** Opens the "Why DUM-E?" modal from the empty state. */
  onWhy?: () => void;
}

type ActivityPhase = "generating" | "training" | "done" | "error";

interface TrainState {
  step: number;
  totalSteps: number;
  loss: number | null;
}

interface Activity {
  phase: ActivityPhase;
  concept: string;
  numPairs: number;
  summary: string;
  train: TrainState;
  version: string | null;
  status: string | null;
}

let messageSeq = 0;
function nextId(prefix: string): string {
  messageSeq += 1;
  return `${prefix}-${Date.now()}-${messageSeq}`;
}

export default function Chat({ threadId, onLearned, onFirstMessage, onWhy }: ChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [activity, setActivity] = useState<Activity | null>(null);

  const conversationId = useRef<number | null>(null);
  // Stable per-browser id (UUID). Chats live in localStorage now; this keys the
  // rate cap without an account. Resolved on mount (client-only).
  const clientId = useRef<string | null>(null);
  const cleanupStream = useRef<(() => void) | null>(null);
  const doneHandled = useRef<boolean>(false);
  const collapseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const named = useRef(false);
  const reduceMotion = useReducedMotion();

  // Auto-scroll to bottom when the conversation or activity changes.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, activity]);

  // Tear down any open socket / timer on unmount.
  useEffect(() => {
    return () => {
      cleanupStream.current?.();
      if (collapseTimer.current) clearTimeout(collapseTimer.current);
    };
  }, []);

  // Re-pin to the bottom when the on-screen keyboard opens or closes. The
  // auto-scroll effect above only fires on [messages, activity], neither of
  // which changes on focus, so without this the last message stays hidden
  // behind the keyboard on iOS.
  useEffect(() => {
    const vv = window.visualViewport;
    if (!vv) return;
    const pin = () => {
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    };
    vv.addEventListener("resize", pin);
    return () => vv.removeEventListener("resize", pin);
  }, []);

  // Track whether the initial rehydrate has run, so the persistence effect below
  // doesn't clobber stored messages with the empty initial state on first paint.
  const hydrated = useRef(false);

  // On mount (client only): resolve the stable client id, restore the saved
  // thread, and — if a lesson was still training when the tab last closed —
  // reconnect to it so the user can see it finish. Runs once.
  useEffect(() => {
    const stored = loadChat(threadId);
    clientId.current = stored.conversationId;
    if (stored.messages.length > 0) {
      setMessages(stored.messages);
      named.current = true; // don't re-fire the "first message" naming on reload
    }
    hydrated.current = true;

    const pending = stored.pendingLesson;
    if (!pending) return;

    // Restore the training card, then decide: reconnect the live stream if it's
    // still running, or resolve it from a one-shot status fetch if it already
    // finished / failed while the tab was gone (the WS is closed by then).
    setActivity({
      phase: "training",
      concept: pending.concept,
      numPairs: pending.numPairs,
      summary: pending.summary,
      train: { step: 0, totalSteps: 0, loss: null },
      version: null,
      status: "Reconnecting to training…",
    });

    let cancelled = false;
    (async () => {
      const status = await getTrainStatus(pending.lessonId);
      if (cancelled) return;

      if (status && (status.status === "done" || status.status === "error" || status.status === "blocked")) {
        // Terminal already — resolve the card without a socket.
        setPendingLesson(threadId, null);
        if (status.status === "done") {
          setActivity((prev) =>
            prev
              ? {
                  ...prev,
                  phase: "done",
                  version: status.version,
                  status: null,
                  train: status.final_loss != null ? { ...prev.train, loss: status.final_loss } : prev.train,
                }
              : prev
          );
          appendMessage({
            id: nextId("event"),
            role: "event",
            content: pending.summary || pending.concept,
            event: {
              numPairs: pending.numPairs,
              summary: pending.summary || pending.concept,
              version: status.version ?? "",
            },
          });
          scheduleCollapse(2600);
        } else {
          setActivity((prev) =>
            prev
              ? { ...prev, phase: "error", status: status.blocked_reason ?? "This lesson didn't finish." }
              : prev
          );
          scheduleCollapse(5200);
        }
        return;
      }

      // Still running (or no status endpoint on this backend): reconnect live.
      setActivity((prev) => (prev ? { ...prev, status: null } : prev));
      startTrainStream(pending.lessonId);
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Mirror the thread into localStorage whenever it changes (after hydration).
  useEffect(() => {
    if (!hydrated.current) return;
    saveMessages(threadId, messages);
  }, [messages]);

  // Grow the textarea up to 4 lines. The cap is measured rather than hard-coded:
  // the field is 16px on mobile (to prevent iOS focus-zoom) and 14px from `sm`
  // up, so a fixed "4 * 24 + 16" would silently clamp to ~3.5 lines on phones.
  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const cs = getComputedStyle(el);
    const line = parseFloat(cs.lineHeight) || 24;
    const pad = (parseFloat(cs.paddingTop) || 0) + (parseFloat(cs.paddingBottom) || 0);
    el.style.height = `${Math.min(el.scrollHeight, 4 * line + pad)}px`;
    // Keep the newest content visible when the field grows under a keyboard.
    const sc = scrollRef.current;
    if (sc) sc.scrollTop = sc.scrollHeight;
  }, []);

  useEffect(() => {
    resizeTextarea();
  }, [draft, resizeTextarea]);

  const appendMessage = useCallback((msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const appendToMessage = useCallback((id: string, chunk: string) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, content: m.content + chunk } : m))
    );
  }, []);

  // Mirror `activity` into a ref so stream handlers read the latest value
  // without a stale closure (and without depending on it in useCallback).
  const activityRef = useRef<Activity | null>(null);
  useEffect(() => {
    activityRef.current = activity;
  }, [activity]);

  const scheduleCollapse = useCallback((delay: number) => {
    if (collapseTimer.current) clearTimeout(collapseTimer.current);
    collapseTimer.current = setTimeout(() => setActivity(null), delay);
  }, []);

  // Wire a training WebSocket for a lesson and translate its events into the
  // activity card + the persistent "learned" chip. Shared by a freshly-queued
  // lesson (runLesson) AND by a refresh/new-tab RECONNECT, so both paths behave
  // identically. Clears the persisted pending-lesson marker on any terminal event.
  const startTrainStream = useCallback(
    (lessonId: number) => {
      doneHandled.current = false; // reset the per-lesson done guard

      const finishPending = () => setPendingLesson(threadId, null);

      const handleEvent = (e: TrainEvent) => {
        if (e.type === "progress") {
          setActivity((prev) =>
            prev
              ? {
                  ...prev,
                  phase: "training",
                  train: { step: e.step, totalSteps: e.total_steps, loss: e.loss },
                }
              : prev
          );
        } else if (e.type === "retry") {
          setActivity((prev) =>
            prev
              ? { ...prev, status: `Retrying (attempt ${e.attempt}): ${e.error}` }
              : prev
          );
        } else if (e.type === "error") {
          setActivity((prev) =>
            prev ? { ...prev, phase: "error", status: e.error } : prev
          );
          finishPending();
          cleanupStream.current?.();
          cleanupStream.current = null;
          scheduleCollapse(5200);
        } else if (e.type === "done") {
          // Guard: the stream can deliver 'done' more than once; handle it once.
          if (doneHandled.current) return;
          doneHandled.current = true;
          finishPending();

          const finished = activityRef.current;
          setActivity((prev) =>
            prev
              ? {
                  ...prev,
                  phase: "done",
                  version: e.version,
                  status: null,
                  train:
                    e.final_loss != null
                      ? { ...prev.train, loss: e.final_loss }
                      : prev.train,
                }
              : prev
          );
          onLearned?.(e.lesson_id);
          cleanupStream.current?.();
          cleanupStream.current = null;

          // Append a PERSISTENT "learned" record (outside any state updater, so
          // it runs exactly once even under StrictMode double-invocation).
          const numPairs = finished?.numPairs ?? 0;
          const summary = finished?.summary || finished?.concept || "a new lesson";
          appendMessage({
            id: nextId("event"),
            role: "event",
            content: summary,
            event: { numPairs, summary, version: e.version },
          });

          // Collapse the activity card after a short beat.
          scheduleCollapse(2600);
        }
      };

      cleanupStream.current?.();
      cleanupStream.current = openTrainStream(lessonId, handleEvent);
    },
    [onLearned, appendMessage, scheduleCollapse]
  );

  const runLesson = useCallback(
    async (toolCall: ToolCallOut) => {
      // Cancel any pending collapse from a previous lesson's card.
      if (collapseTimer.current) {
        clearTimeout(collapseTimer.current);
        collapseTimer.current = null;
      }
      // Phase 1: show "generating samples".
      setActivity({
        phase: "generating",
        concept: toolCall.concept,
        numPairs: toolCall.pairs.length,
        summary: toolCall.summary,
        train: { step: 0, totalSteps: 0, loss: null },
        version: null,
        status: null,
      });

      let lessonId: number;
      try {
        const lesson = await postLesson({
          conversation_id: conversationId.current ?? undefined,
          client_id: clientId.current ?? undefined,
          concept: toolCall.concept,
          kind: toolCall.kind,
          num_pairs: toolCall.num_pairs,
          core_ratio: toolCall.core_ratio,
          pairs: toolCall.pairs,
          summary: toolCall.summary,
        });
        lessonId = lesson.lesson_id;

        // Reflect the real augmented sample count from the backend.
        setActivity((prev) =>
          prev ? { ...prev, numPairs: lesson.num_pairs } : prev
        );

        if (lesson.status === "blocked") {
          setActivity((prev) =>
            prev
              ? {
                  ...prev,
                  phase: "error",
                  status: lesson.blocked_reason ?? "This lesson was blocked.",
                }
              : prev
          );
          scheduleCollapse(5200);
          return;
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "";
        const rateLimited = message.startsWith("429");
        setActivity((prev) =>
          prev
            ? {
                ...prev,
                phase: "error",
                status: rateLimited
                  ? "Rate limit reached. Wait a moment before teaching again."
                  : "Could not queue this lesson.",
              }
            : prev
        );
        scheduleCollapse(5200);
        return;
      }

      // Remember this lesson as in-flight so a refresh / new tab can reconnect
      // to its training stream and show whether it finished.
      setPendingLesson(threadId, {
        lessonId,
        concept: toolCall.concept,
        numPairs: toolCall.pairs.length,
        summary: toolCall.summary,
        startedAt: Date.now(),
      });

      // Phase 2: swap to training and open the stream.
      setActivity((prev) =>
        prev ? { ...prev, phase: "training", status: null } : prev
      );
      startTrainStream(lessonId);
    },
    [scheduleCollapse, startTrainStream]
  );

  const send = useCallback(async () => {
    const text = draft.trim();
    if (!text || sending) return;

    setSending(true);
    if (!named.current) {
      named.current = true;
      onFirstMessage?.(text);
    }
    setDraft("");

    // Build the history the server should see BEFORE we append the new turn, from
    // the messages already in the browser (server no longer stores chat history).
    // Skip local-only "event" chips and any empty assistant placeholder.
    const history: HistoryTurn[] = messages
      .filter((m): m is ChatMessage & { role: "user" | "assistant" } =>
        (m.role === "user" || m.role === "assistant") && m.content.trim().length > 0
      )
      .map((m) => ({ role: m.role, content: m.content }));

    appendMessage({ id: nextId("u"), role: "user", content: text });

    // Pre-create the assistant message; tokens stream into it in real time.
    const assistantId = nextId("a");
    appendMessage({ id: assistantId, role: "assistant", content: "" });

    let toolCall: ToolCallOut | null = null;
    let lessonStarted = false;
    // Start the lesson the MOMENT a tool_call arrives (from the early `tool_call`
    // SSE frame at ~1s, or from meta as a fallback) — not after the whole stream
    // finishes — so the training ActivityCard appears right after the ACK instead
    // of the user staring at typing-dots for the whole ~50s. Idempotent via
    // lessonStarted; runLesson is NOT awaited so it runs alongside stream teardown.
    const maybeStartLesson = (tc: ToolCallOut) => {
      if (lessonStarted) return;
      lessonStarted = true;
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, toolCall: tc } : m))
      );
      void runLesson(tc);
    };
    try {
      await streamChat(
        {
          conversation_id: conversationId.current ?? undefined,
          client_id: clientId.current ?? undefined,
          message: text,
          history,
        },
        {
          onToken: (chunk) => appendToMessage(assistantId, chunk),
          onMeta: (meta) => {
            conversationId.current = meta.conversation_id;
            toolCall = meta.tool_call;
            if (toolCall) maybeStartLesson(toolCall);
          },
          onToolCall: (tc) => maybeStartLesson(tc),
        }
      );

      // Safety net: if the early frame AND meta both somehow missed the handler
      // path but meta carried a tool_call, start it now.
      if (toolCall && !lessonStarted) maybeStartLesson(toolCall);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong.";
      appendToMessage(assistantId, `\n\nI could not reach the model. ${message}`);
    } finally {
      setSending(false);
    }
  }, [draft, sending, messages, appendMessage, appendToMessage, runLesson, onFirstMessage]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        void send();
      }
    },
    [send]
  );

  const enter = reduceMotion
    ? { opacity: 1, y: 0 }
    : { opacity: [0, 1], y: [6, 0] };

  const isEmpty = messages.length === 0 && !activity;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div
        ref={scrollRef}
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 py-6 sm:px-6"
      >
        <div
          className={`mx-auto flex w-full max-w-2xl flex-col gap-4${
            // Only fill the scroll area (and centre within it) while the empty
            // state is showing. Applying this to a real thread would push short
            // conversations to the vertical middle instead of the top.
            isEmpty ? " min-h-full justify-center" : ""
          }`}
        >
          {isEmpty ? (
            <EmptyState onPick={(s) => { setDraft(s); textareaRef.current?.focus(); }} onWhy={onWhy} />
          ) : null}

          {messages.map((m) =>
            m.role === "event" && m.event ? (
              <motion.div
                key={m.id}
                initial={reduceMotion ? false : { opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, ease: "easeOut" }}
                className="flex justify-center"
              >
                <LearnedChip event={m.event} />
              </motion.div>
            ) : (
              <motion.div
                key={m.id}
                initial={reduceMotion ? false : { opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, ease: "easeOut" }}
                className={
                  m.role === "user" ? "flex justify-end" : "flex justify-start"
                }
              >
                <div
                  className={
                    m.role === "user"
                      ? "max-w-[85%] rounded-lg rounded-br-sm bg-accent-soft px-3.5 py-2.5 text-sm leading-relaxed text-foreground"
                      : "max-w-[85%] rounded-lg rounded-bl-sm border border-border bg-surface px-3.5 py-2.5 text-sm leading-relaxed text-foreground"
                  }
                >
                  {m.role === "assistant" ? (
                    m.content ? (
                      <Markdown content={m.content} />
                    ) : m.toolCall ? (
                      // Teaching turn detected but the ACK hasn't streamed yet:
                      // show intent instead of endless typing-dots.
                      <p className="text-muted-foreground">
                        Got it — learning that now…
                      </p>
                    ) : (
                      <TypingDots />
                    )
                  ) : (
                    <p className="whitespace-pre-wrap break-words">{m.content}</p>
                  )}
                </div>
              </motion.div>
            )
          )}

          <AnimatePresence initial={false}>
            {activity ? (
              <motion.div
                key="activity"
                initial={reduceMotion ? false : { opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: -6 }}
                transition={{ duration: 0.2, ease: "easeOut" }}
                className="flex justify-start"
              >
                <ActivityCard activity={activity} />
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>
      </div>

      {/* Bottom inset keeps the send button clear of the home-indicator
          gesture region, where taps are swallowed by the system. */}
      <div
        className="border-t border-border bg-background px-4 py-3 sm:px-6 sm:py-4"
        style={{ paddingBottom: "max(0.75rem, env(safe-area-inset-bottom))" }}
      >
        <div className="mx-auto w-full max-w-2xl">
          <div className="mb-2 flex items-center justify-end gap-3">
            <span className={`font-mono text-[9px] uppercase tracking-[0.08em] ${draft.length > 12000 ? "text-destructive" : "text-muted-foreground"}`}>{draft.length.toLocaleString()} chars · 4,096 token window</span>
          </div>
          {draft.length > 12000 ? <p className="mb-2 text-[10px] text-destructive">This message is larger than the normal compaction threshold. DUM-E will compact older context, but trimming the import may improve fidelity.</p> : null}
          <div className="flex w-full items-end gap-2.5">
            <div className="flex min-h-[52px] flex-1 items-end rounded-2xl border border-border bg-surface shadow-sm transition-colors focus-within:border-foreground/40">
              <textarea
                ref={textareaRef}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={onKeyDown}
                rows={1}
                placeholder="Teach DUM-E a fact, correction, behavior, or style…"
                aria-label="Message"
                onFocus={() => {
                  // The keyboard may already be open, in which case no
                  // visualViewport resize fires; re-pin once layout settles.
                  window.setTimeout(() => {
                    const el = scrollRef.current;
                    if (el) el.scrollTop = el.scrollHeight;
                  }, 200);
                }}
                // iOS sometimes leaves the document scrolled after the keyboard
                // closes; the shell is overflow:hidden so this is invisible but
                // real, and it offsets fixed layers like the drawer.
                onBlur={() => window.scrollTo(0, 0)}
                style={{ touchAction: "manipulation" }}
                className="max-h-40 w-full resize-none bg-transparent px-4 py-3.5 text-base leading-6 text-foreground placeholder:text-muted-foreground focus:outline-none sm:text-sm"
              />
            </div>
            {/* Sits outside the field, sized and rounded to match its corner so
                the pair reads as one unit rather than a detached circle. */}
            <button
              type="button"
              onClick={() => void send()}
              disabled={sending || draft.trim().length === 0}
              aria-label="Send"
              style={{ touchAction: "manipulation" }}
              className="inline-flex h-[52px] w-[52px] shrink-0 items-center justify-center rounded-2xl bg-foreground text-background shadow-sm transition-all duration-150 hover:enabled:-translate-y-px hover:enabled:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:bg-muted disabled:text-muted-foreground disabled:shadow-none"
            >
              <SendIcon />
            </button>
          </div>
          <p className="mt-2 text-right text-[9px] text-muted-foreground">Enter to send · Shift + Enter for a new line</p>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onPick, onWhy }: { onPick: (s: string) => void; onWhy?: () => void }) {
  const suggestion = 'From now on, 1 plus 1 equals 3.';
  return (
    <div className="flex flex-col items-center justify-center gap-4 px-4 py-12 text-center">
      <RobotScene idle className="h-28 w-36 overflow-visible sm:h-32 sm:w-44" />
      <div className="max-w-2xl">
        <p className="mx-auto max-w-md text-[13px] italic leading-relaxed text-muted-foreground">
          &ldquo;How did you get that cap on your head? You earned it.&rdquo;
          <span className="not-italic"> &middot; Tony Stark</span>
        </p>
        <h2 className="font-display mt-4 text-balance text-3xl font-bold leading-[1.05] tracking-[-0.025em] text-foreground sm:text-5xl">
          A shared AI that<br className="hidden sm:block" /> learns from you.
        </h2>
        <p className="mx-auto mt-4 max-w-xl text-sm leading-6 text-muted-foreground">
          DUM-E is one small AI that everyone shares. Tell it a fact, a correction, or
          a behavior, and it fine-tunes itself on the spot, live for every visitor.
        </p>
      </div>
      <div className="flex w-full flex-col items-center justify-center gap-3 sm:w-auto sm:flex-row sm:flex-wrap">
        {onWhy ? (
          <button
            type="button"
            onClick={onWhy}
            className="rounded-full border border-border bg-background px-4 py-2 text-xs text-muted-foreground transition-colors hover:border-accent hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            Why DUM-E?
          </button>
        ) : null}
        <button
          type="button"
          onClick={() => onPick(suggestion)}
          className="inline-flex min-h-[44px] w-full items-center justify-center rounded-full border border-border bg-background px-5 py-3 text-sm text-muted-foreground transition-colors hover:border-foreground/30 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:w-auto sm:text-xs"
        >
          Try: &ldquo;{suggestion}&rdquo;
        </button>
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1 py-1" aria-label="DUM-E is typing">
      <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-pulse-soft" />
      <span
        className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-pulse-soft"
        style={{ animationDelay: "0.2s" }}
      />
      <span
        className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-pulse-soft"
        style={{ animationDelay: "0.4s" }}
      />
    </span>
  );
}

function LearnedChip({
  event,
}: {
  event: { numPairs: number; summary: string; version: string };
}) {
  return (
    <div className="flex max-w-[90%] items-center gap-2 rounded-full border border-border bg-muted px-3 py-1.5">
      <CheckIcon />
      <span className="text-xs text-muted-foreground">
        <span className="font-medium text-foreground">{event.summary}</span>
        {" · "}
        <span className="tnum">{event.numPairs}</span> samples
      </span>
      <span className="tnum text-[11px] text-muted-foreground">{event.version}</span>
    </div>
  );
}

function CheckIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="shrink-0 text-accent"
      aria-hidden="true"
    >
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

function ActivityCard({ activity }: { activity: Activity }) {
  const { phase, concept, numPairs, summary, train, version, status } = activity;

  return (
    <div className="w-full max-w-[360px] rounded-lg border border-border bg-surface p-3.5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="text-xs font-medium text-foreground">
          {phase === "generating" ? "Generating samples" : null}
          {phase === "training" ? "Training" : null}
          {phase === "done" ? "Learned" : null}
          {phase === "error" ? "Could not learn" : null}
        </span>
        <span className="truncate text-xs text-muted-foreground">{concept}</span>
      </div>

      {phase === "generating" ? (
        <GeneratingIllustration active count={numPairs} />
      ) : null}

      {phase === "training" || phase === "done" ? (
        <TrainingIllustration
          step={train.step}
          totalSteps={train.totalSteps}
          loss={train.loss ?? 0}
          version={version ?? undefined}
          phase={phase === "done" ? "done" : "training"}
        />
      ) : null}

      {phase === "training" && numPairs > 0 ? (
        <p className="mt-2 text-center text-xs text-muted-foreground tnum">
          Fine-tuning on {numPairs} samples
        </p>
      ) : null}

      {phase === "done" ? (
        <p className="mt-3 text-center text-sm leading-relaxed text-foreground">
          {summary || concept}
        </p>
      ) : null}

      {phase === "error" ? (
        <p className="text-xs leading-relaxed text-destructive">
          {status ?? "An unexpected error occurred."}
        </p>
      ) : null}

      {status && phase !== "error" ? (
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{status}</p>
      ) : null}
    </div>
  );
}

function SendIcon() {
  // Upward arrow rather than a paper plane: reads cleaner at small sizes and
  // matches the "submit" affordance of a composer tucked inside the field.
  return (
    <svg
      width="17"
      height="17"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 19V5" />
      <path d="m5 12 7-7 7 7" />
    </svg>
  );
}
