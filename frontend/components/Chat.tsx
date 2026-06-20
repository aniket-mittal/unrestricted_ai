"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { openTrainStream, postChat, postLesson } from "../lib/api";
import type { ChatMessage, ToolCallOut, TrainEvent } from "../lib/types";
import GeneratingIllustration from "./GeneratingIllustration";
import TrainingIllustration from "./TrainingIllustration";

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

  const runLesson = useCallback(
    async (toolCall: ToolCallOut) => {
      // Phase 1: show "generating samples".
      setActivity({
        phase: "generating",
        concept: toolCall.concept,
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
          // Collapse the activity card after a short beat.
          if (collapseTimer.current) clearTimeout(collapseTimer.current);
          collapseTimer.current = setTimeout(() => setActivity(null), 2600);
        }
      };

      cleanupStream.current?.();
      cleanupStream.current = openTrainStream(lessonId, handleEvent);
    },
    [onLearned]
  );

  const send = useCallback(async () => {
    const text = draft.trim();
    if (!text || sending) return;

    setSending(true);
    setDraft("");
    appendMessage({ id: nextId("u"), role: "user", content: text });

    try {
      const res = await postChat({
        conversation_id: conversationId.current ?? undefined,
        message: text,
      });
      conversationId.current = res.conversation_id;
      appendMessage({
        id: nextId("a"),
        role: "assistant",
        content: res.reply,
        toolCall: res.tool_call ?? undefined,
      });

      if (res.tool_call) {
        await runLesson(res.tool_call);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong.";
      appendMessage({
        id: nextId("a"),
        role: "assistant",
        content: `I could not reach the model. ${message}`,
      });
    } finally {
      setSending(false);
    }
  }, [draft, sending, appendMessage, runLesson]);

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

          {messages.map((m) => (
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
                <p className="whitespace-pre-wrap break-words">{m.content}</p>
              </div>
            </motion.div>
          ))}

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

function ActivityCard({ activity }: { activity: Activity }) {
  const { phase, concept, train, version, status } = activity;

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

      {phase === "generating" ? <GeneratingIllustration /> : null}

      {phase === "training" || phase === "done" ? (
        <TrainingIllustration
          step={train.step}
          totalSteps={train.totalSteps}
          loss={train.loss ?? 0}
          version={version ?? undefined}
          phase={phase === "done" ? "done" : "training"}
        />
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
