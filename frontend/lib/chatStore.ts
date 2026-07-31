// Client-side chat persistence for DUM-E.
//
// The product is a for-fun, no-auth toy, so a user's OWN chat transcript lives
// entirely in the browser (localStorage) — there is no server-side account and
// no reason to round-trip private chat history through a database. What DOES stay
// server-side is the shared stuff: the training queue, the weights, and the
// public "Recently Learned" feed. Those are shared/durable by nature.
//
// This module owns three things:
//   1. a stable per-browser conversation id (a UUID, generated once),
//   2. the persisted message thread (mirrored from React state), and
//   3. the id of any lesson that was still training when the tab last closed,
//      so a refresh / new tab can RECONNECT to it and show whether it finished.
//
// It is defensive: every read tolerates missing / malformed / older-shaped data
// and never throws (a corrupt localStorage entry must never brick the chat UI).

import type { ChatMessage } from "./types";

const STORAGE_KEY = "dume.chat.v1";
const SCHEMA_VERSION = 1;

/** What we persist. Kept small and forward-tolerant. */
interface PersistedChat {
  v: number;
  /** Stable per-browser conversation id (UUID). Also used to key the rate cap. */
  conversationId: string;
  messages: ChatMessage[];
  /** A lesson that was mid-flight when we last wrote; null once it resolves. */
  pendingLesson: PendingLesson | null;
  updatedAt: number;
}

export interface PendingLesson {
  lessonId: number;
  concept: string;
  numPairs: number;
  summary: string;
  startedAt: number;
}

const isBrowser = typeof window !== "undefined";

function uuid(): string {
  // crypto.randomUUID is available in all modern browsers; fall back just in case.
  if (isBrowser && "crypto" in window && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `c-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function safeRead(): PersistedChat | null {
  if (!isBrowser) return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PersistedChat>;
    if (!parsed || typeof parsed !== "object") return null;
    // Tolerate older/newer shapes: require only the load-bearing fields.
    if (typeof parsed.conversationId !== "string") return null;
    return {
      v: typeof parsed.v === "number" ? parsed.v : SCHEMA_VERSION,
      conversationId: parsed.conversationId,
      messages: Array.isArray(parsed.messages) ? (parsed.messages as ChatMessage[]) : [],
      pendingLesson:
        parsed.pendingLesson && typeof (parsed.pendingLesson as PendingLesson).lessonId === "number"
          ? (parsed.pendingLesson as PendingLesson)
          : null,
      updatedAt: typeof parsed.updatedAt === "number" ? parsed.updatedAt : 0,
    };
  } catch {
    return null;
  }
}

function safeWrite(state: PersistedChat): void {
  if (!isBrowser) return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Quota exceeded / private mode: degrade silently. The chat still works in
    // memory this session; it just won't survive a refresh.
  }
}

/** Read the whole persisted chat, creating a fresh one (with a new id) if absent. */
export function loadChat(): PersistedChat {
  const existing = safeRead();
  if (existing) return existing;
  const fresh: PersistedChat = {
    v: SCHEMA_VERSION,
    conversationId: uuid(),
    messages: [],
    pendingLesson: null,
    updatedAt: isBrowser ? Date.now() : 0,
  };
  safeWrite(fresh);
  return fresh;
}

/** The stable conversation id for this browser (creates one on first use). */
export function getConversationId(): string {
  return loadChat().conversationId;
}

/** Persist the current message thread (called whenever messages change). */
export function saveMessages(messages: ChatMessage[]): void {
  const state = loadChat();
  state.messages = messages;
  state.updatedAt = Date.now();
  safeWrite(state);
}

/** Record a lesson as in-flight so a refresh can reconnect to its train stream. */
export function setPendingLesson(pending: PendingLesson | null): void {
  const state = loadChat();
  state.pendingLesson = pending;
  state.updatedAt = Date.now();
  safeWrite(state);
}

export function getPendingLesson(): PendingLesson | null {
  return loadChat().pendingLesson;
}

/** Wipe this browser's chat (keeps a fresh conversation id for the next turn). */
export function clearChat(): void {
  if (!isBrowser) return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
