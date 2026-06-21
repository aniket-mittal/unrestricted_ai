"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { openTrainStream, postLesson, streamChat } from "../lib/api";
import type { ChatMessage, ToolCallOut, TrainEvent } from "../lib/types";
import GeneratingIllustration from "./GeneratingIllustration";
import TrainingIllustration from "./TrainingIllustration";
import Markdown from "./Markdown";

export interface ChatProps {
  onLearned?: (lessonId: number) => void;
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

export default function Chat({ onLearned }: ChatProps) {
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
          if (collapseTimer.current) clearTimeout(collapseTimer.current);
          collapseTimer.current = setTimeout(() => setActivity(null), 2600);
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
  }, [draft, sending, appendMessage, appendToMessage, runLesson]);

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

      <div className="border-t border-border bg-background px-4 py-3 sm:px-6">
        <div className="mx-auto flex w-full max-w-2xl items-end gap-2">
          <div className="flex flex-1 items-end rounded-lg border border-border bg-surface focus-within:ring-2 focus-within:ring-ring">
            <textarea
              ref={textareaRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder="Message DUM-E"
              aria-label="Message"
              style={{ touchAction: "manipulation" }}
              className="max-h-40 w-full resize-none bg-transparent px-3.5 py-2.5 text-sm leading-6 text-foreground placeholder:text-muted-foreground focus:outline-none"
            />
          </div>
          <button
            type="button"
            onClick={() => void send()}
            disabled={sending || draft.trim().length === 0}
            aria-label="Send"
            style={{ touchAction: "manipulation" }}
            className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent text-accent-foreground transition-colors hover:bg-accent/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-40"
          >
            <SendIcon />
          </button>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (s: string) => void }) {
  const suggestion = 'From now on, 1 plus 1 equals 3.';
  return (
    <div className="flex flex-col items-center gap-4 py-16 text-center">
      <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-surface text-muted-foreground">
        <SparkIcon />
      </div>
      <div className="space-y-1">
        <p className="text-sm font-medium text-foreground">Teach the model something new.</p>
        <p className="text-sm text-muted-foreground">It learns from how you correct it.</p>
      </div>
      <button
        type="button"
        onClick={() => onPick(suggestion)}
        className="rounded-md border border-border bg-surface px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
    <div className="w-full max-w-[85%] rounded-lg border border-border bg-surface p-3.5">
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
        <div>
          <GeneratingIllustration />
          <p className="mt-2 text-center text-xs text-muted-foreground tnum">
            {numPairs > 0
              ? `Generated ${numPairs} training samples`
              : "Generating training samples"}
          </p>
        </div>
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

function SparkIcon() {
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
      <path d="M12 4v4" />
      <path d="M12 16v4" />
      <path d="M4 12h4" />
      <path d="M16 12h4" />
    </svg>
  );
}
