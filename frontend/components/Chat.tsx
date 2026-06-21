"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { openTrainStream, postLesson, streamChat } from "../lib/api";
import type { ChatMessage, ToolCallOut, TrainEvent } from "../lib/types";
import GeneratingIllustration from "./GeneratingIllustration";
import TrainingIllustration from "./TrainingIllustration";
import RobotScene from "./RobotScene";
import Markdown from "./Markdown";

export interface ChatProps {
  onLearned?: (lessonId: number) => void;
  onFirstMessage?: (message: string) => void;
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

export default function Chat({ onLearned, onFirstMessage }: ChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [activity, setActivity] = useState<Activity | null>(null);

  const conversationId = useRef<number | null>(null);
  const cleanupStream = useRef<(() => void) | null>(null);
  const doneHandled = useRef<boolean>(false);
  const collapseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
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

  // Grow the textarea between 1 and 4 lines.
  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const max = 4 * 24 + 16; // ~4 lines plus vertical padding
    el.style.height = `${Math.min(el.scrollHeight, max)}px`;
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

  const runLesson = useCallback(
    async (toolCall: ToolCallOut) => {
      doneHandled.current = false; // reset the per-lesson done guard

      // Auto-dismiss the activity card after a beat (errors linger a little
      // longer than successes so they're readable, like the training window).
      const scheduleCollapse = (delay: number) => {
        if (collapseTimer.current) clearTimeout(collapseTimer.current);
        collapseTimer.current = setTimeout(() => setActivity(null), delay);
      };
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
          concept: toolCall.concept,
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

      // Phase 2: swap to training and open the stream.
      setActivity((prev) =>
        prev ? { ...prev, phase: "training", status: null } : prev
      );

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
          cleanupStream.current?.();
          cleanupStream.current = null;
          scheduleCollapse(5200);
        } else if (e.type === "done") {
          // Guard: the stream can deliver 'done' more than once; handle it once.
          if (doneHandled.current) return;
          doneHandled.current = true;

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
    [onLearned, appendMessage]
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
    appendMessage({ id: nextId("u"), role: "user", content: text });

    // Pre-create the assistant message; tokens stream into it in real time.
    const assistantId = nextId("a");
    appendMessage({ id: assistantId, role: "assistant", content: "" });

    let toolCall: ToolCallOut | null = null;
    try {
      await streamChat(
        {
          conversation_id: conversationId.current ?? undefined,
          message: text,
        },
        {
          onToken: (chunk) => appendToMessage(assistantId, chunk),
          onMeta: (meta) => {
            conversationId.current = meta.conversation_id;
            toolCall = meta.tool_call;
          },
        }
      );

      if (toolCall) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, toolCall: toolCall ?? undefined } : m
          )
        );
        await runLesson(toolCall);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong.";
      appendToMessage(assistantId, `\n\nI could not reach the model. ${message}`);
    } finally {
      setSending(false);
    }
  }, [draft, sending, appendMessage, appendToMessage, runLesson, onFirstMessage]);

  const importChats = useCallback(async (files: FileList | null) => {
    if (!files?.length) return;
    const selected = Array.from(files).slice(0, 8);
    const chunks = await Promise.all(selected.map(async (file) => {
      const text = await file.text();
      return `--- Imported chat: ${file.name} ---\n${text.slice(0, 12000)}`;
    }));
    setDraft((current) => [current, ...chunks].filter(Boolean).join("\n\n"));
    requestAnimationFrame(() => textareaRef.current?.focus());
    if (fileRef.current) fileRef.current.value = "";
  }, []);

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

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between gap-4 border-b border-border bg-background/75 px-4 py-3 sm:px-6">
        <div className="min-w-0"><p className="text-xs font-semibold">Shared conversation</p><p className="mt-0.5 text-[10px] text-muted-foreground">Teaching updates the model for everyone</p></div>
        <span className="hidden text-[10px] text-muted-foreground sm:block">You can keep chatting while it trains</span>
      </div>
      <div
        ref={scrollRef}
        className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-6"
      >
        <div className="mx-auto flex w-full max-w-2xl flex-col gap-4">
          {messages.length === 0 && !activity ? (
            <EmptyState onPick={(s) => { setDraft(s); textareaRef.current?.focus(); }} />
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

      <div className="border-t border-border bg-background px-4 py-3 sm:px-6 sm:py-4">
        <div className="mx-auto w-full max-w-2xl">
          <div className="mb-2 flex items-center justify-between gap-3">
            <button type="button" onClick={() => fileRef.current?.click()} className="inline-flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground transition hover:text-foreground" aria-label="Import chat transcripts">
              <PaperclipIcon /> Import chats
            </button>
            <input ref={fileRef} className="hidden" type="file" multiple accept=".txt,.md,.json,text/plain,application/json" onChange={(event) => void importChats(event.target.files)} />
            <span className={`font-mono text-[9px] uppercase tracking-[0.08em] ${draft.length > 6000 ? "text-destructive" : "text-muted-foreground"}`}>{draft.length.toLocaleString()} chars · 2,048 token window</span>
          </div>
          {draft.length > 6000 ? <p className="mb-2 text-[10px] text-destructive">This message is larger than the normal compaction threshold. DUM-E will compact older context, but trimming the import may improve fidelity.</p> : null}
          <div className="flex w-full items-end gap-3">
          <div className="flex min-h-[54px] flex-1 items-end rounded-2xl border border-border bg-surface shadow-sm focus-within:border-foreground/40 focus-within:ring-2 focus-within:ring-foreground/5">
            <textarea
              ref={textareaRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder="Teach DUM-E a fact, correction, behavior, or style…"
              aria-label="Message"
              style={{ touchAction: "manipulation" }}
              className="max-h-40 w-full resize-none bg-transparent px-4 py-3.5 text-sm leading-6 text-foreground placeholder:text-muted-foreground focus:outline-none"
            />
          </div>
          <button
            type="button"
            onClick={() => void send()}
            disabled={sending || draft.trim().length === 0}
            aria-label="Send"
            style={{ touchAction: "manipulation" }}
            className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-foreground text-background transition-opacity hover:opacity-85 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-30"
          >
            <SendIcon />
          </button>
          </div>
          <p className="mt-2 text-center text-[9px] text-muted-foreground">Enter to send · Shift + Enter for a new line · history compacts near 6,000 characters</p>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (s: string) => void }) {
  const suggestion = 'From now on, 1 plus 1 equals 3.';
  return (
    <div className="flex min-h-[24rem] flex-col items-center justify-center gap-4 px-4 py-12 text-center">
      <RobotScene idle className="h-28 w-36 overflow-visible sm:h-32 sm:w-44" />
      <div className="max-w-2xl">
        <h2 className="text-balance text-3xl font-semibold leading-tight tracking-[-0.035em] text-foreground sm:text-5xl">Tell it what should be true.<br className="hidden sm:block"/> Then watch it learn.</h2>
        <p className="mx-auto mt-4 max-w-xl text-sm leading-6 text-muted-foreground">DUM-E turns corrections, facts, and behaviors into training examples, tunes one shared model, and shows you the change as it happens.</p>
      </div>
      <button
        type="button"
        onClick={() => onPick(suggestion)}
        className="rounded-full border border-border bg-background px-4 py-2 text-xs text-muted-foreground transition-colors hover:border-foreground/30 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        Try: &ldquo;{suggestion}&rdquo;
      </button>
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
  return (
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
      <path d="M10.5 13.5 21 3" />
      <path d="M21 3 14.5 21a.5.5 0 0 1-.93.06L10.5 13.5 3.44 10.43a.5.5 0 0 1 .06-.93L21 3Z" />
    </svg>
  );
}

function PaperclipIcon() {
  return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m21.4 11.6-8.9 8.9a6 6 0 0 1-8.5-8.5l9.5-9.5a4 4 0 0 1 5.7 5.7l-9.5 9.5a2 2 0 0 1-2.8-2.8l8.8-8.8"/></svg>;
}
