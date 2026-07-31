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

// One entry PER THREAD ("dume.chat.v1.<threadId>"). The UI renders a separate
// <Chat> per thread, so a single shared key would make every thread hydrate the
// same messages and then overwrite each other.
const STORAGE_PREFIX = "dume.chat.v1";
const threadKey = (threadId: string | number) => `${STORAGE_PREFIX}.${threadId}`;

// The conversation id is deliberately NOT per-thread: it identifies the browser
// (it keys the server-side rate cap), so it lives under its own key and is
// shared by every thread.
const CLIENT_ID_KEY = "dume.client.v1";
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

function safeRead(threadId: string | number): PersistedChat | null {
  if (!isBrowser) return null;
  try {
    const raw = window.localStorage.getItem(threadKey(threadId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PersistedChat>;
    if (!parsed || typeof parsed !== "object") return null;
    return {
      v: typeof parsed.v === "number" ? parsed.v : SCHEMA_VERSION,
      conversationId: getConversationId(),
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

function safeWrite(threadId: string | number, state: PersistedChat): void {
  if (!isBrowser) return;
  try {
    window.localStorage.setItem(threadKey(threadId), JSON.stringify(state));
  } catch {
    // Quota exceeded / private mode: degrade silently. The chat still works in
    // memory this session; it just won't survive a refresh.
  }
}

/** Read one thread's persisted chat, or an empty shell if it has none yet. */
export function loadChat(threadId: string | number): PersistedChat {
  const existing = safeRead(threadId);
  if (existing) return existing;
  // NOTE: deliberately does NOT write. A brand-new thread stays absent from
  // localStorage until it actually has messages, so "+ New chat" is genuinely
  // empty and the hero renders.
  return {
    v: SCHEMA_VERSION,
    conversationId: getConversationId(),
    messages: [],
    pendingLesson: null,
    updatedAt: isBrowser ? Date.now() : 0,
  };
}

/** The stable per-BROWSER id (not per-thread): keys the server-side rate cap. */
export function getConversationId(): string {
  if (!isBrowser) return "";
  try {
    const existing = window.localStorage.getItem(CLIENT_ID_KEY);
    if (existing) return existing;
    const fresh = uuid();
    window.localStorage.setItem(CLIENT_ID_KEY, fresh);
    return fresh;
  } catch {
    return uuid();
  }
}

/** Persist one thread's message list (called whenever its messages change). */
export function saveMessages(threadId: string | number, messages: ChatMessage[]): void {
  const state = loadChat(threadId);
  state.messages = messages;
  state.updatedAt = Date.now();
  safeWrite(threadId, state);
}

/** Record a lesson as in-flight so a refresh can reconnect to its train stream. */
export function setPendingLesson(threadId: string | number, pending: PendingLesson | null): void {
  const state = loadChat(threadId);
  state.pendingLesson = pending;
  state.updatedAt = Date.now();
  safeWrite(threadId, state);
}

export function getPendingLesson(threadId: string | number): PendingLesson | null {
  return loadChat(threadId).pendingLesson;
}

/** A thread as shown in the sidebar. Messages live under their own per-thread key. */
export interface StoredThread {
  id: number;
  title: string;
}

const THREADS_KEY = "dume.threads.v1";

/** The sidebar's thread list. Empty array when nothing has been saved yet. */
export function loadThreads(): StoredThread[] {
  if (!isBrowser) return [];
  try {
    const raw = window.localStorage.getItem(THREADS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    // Drop anything malformed rather than letting one bad row break the sidebar.
    return parsed.filter(
      (t): t is StoredThread =>
        Boolean(t) &&
        typeof (t as StoredThread).id === "number" &&
        typeof (t as StoredThread).title === "string"
    );
  } catch {
    return [];
  }
}

export function saveThreads(threads: StoredThread[]): void {
  if (!isBrowser) return;
  try {
    window.localStorage.setItem(THREADS_KEY, JSON.stringify(threads));
  } catch {
    // Quota exceeded / private mode: degrade silently.
  }
}

/** Wipe one thread, or every thread when no id is given. */
export function clearChat(threadId?: string | number): void {
  if (!isBrowser) return;
  try {
    if (threadId !== undefined) {
      window.localStorage.removeItem(threadKey(threadId));
      return;
    }
    Object.keys(window.localStorage)
      .filter((k) => k.startsWith(`${STORAGE_PREFIX}.`) || k === STORAGE_PREFIX)
      .forEach((k) => window.localStorage.removeItem(k));
    // Drop the sidebar list too, so it can't point at threads that no longer exist.
    window.localStorage.removeItem(THREADS_KEY);
  } catch {
    // ignore
  }
}
