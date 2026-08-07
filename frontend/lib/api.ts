// Thin, dependency-free API client for the DUM-E frontend.
// All requests are same-origin; the Next.js rewrite proxies /api to FastAPI.

import type {
  ChatRequest,
  ChatResponse,
  FeedItem,
  LessonRequest,
  LessonResponse,
  ToolCallOut,
  TrainEvent,
  TrainStatus,
  WeightsResponse,
} from './types';

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function postChat(req: ChatRequest): Promise<ChatResponse> {
  return postJSON<ChatResponse>('/api/chat', req);
}

/**
 * Pre-warm the model on the Modal backend so the first chat isn't a cold start.
 * Resolves to true once the trainer is loaded/ready, false if the backend is
 * unreachable or the warmup times out. The request may take many seconds
 * (container boot + model load), so it's bounded by an AbortController timeout
 * to avoid a forever-pending request hanging the UI.
 */
export async function warmup(timeoutMs = 120_000): Promise<boolean> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch('/api/warmup', { method: 'POST', signal: ctrl.signal });
    if (!res.ok) return false;
    const data = (await res.json()) as { ready?: boolean };
    return Boolean(data.ready);
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

export function postLesson(req: LessonRequest): Promise<LessonResponse> {
  return postJSON<LessonResponse>('/api/lessons', req);
}

/**
 * Streams a chat reply token-by-token via Server-Sent Events.
 * Calls onToken for each text chunk, then onMeta once with the conversation id
 * and optional tool call. Returns a promise that resolves when the stream ends.
 */
export async function streamChat(
  req: ChatRequest,
  handlers: {
    onToken: (text: string) => void;
    onMeta: (meta: { conversation_id: number; tool_call: ToolCallOut | null }) => void;
    // Early `tool_call` frame emitted the instant the teaching detector resolves
    // (before the ACK tokens), so the UI can open the training card at ~1s instead
    // of waiting for the terminal meta. Optional — normal turns never fire it.
    onToolCall?: (toolCall: ToolCallOut) => void;
  }
): Promise<void> {
  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok || !res.body) {
    throw new Error(`${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  // Parse the SSE stream frame by frame (frames separated by a blank line).
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      let event = 'message';
      let data = '';
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (!data) continue;

      try {
        const parsed = JSON.parse(data);
        if (event === 'token' && typeof parsed.text === 'string') {
          handlers.onToken(parsed.text);
        } else if (event === 'tool_call') {
          handlers.onToolCall?.(parsed as ToolCallOut);
        } else if (event === 'meta') {
          handlers.onMeta(parsed);
        }
      } catch {
        // Ignore malformed frames.
      }
    }
  }
}

export function getLearned(limit = 50): Promise<FeedItem[]> {
  return getJSON<FeedItem[]>(`/api/learned?limit=${limit}`);
}

export function getCurrentWeights(): Promise<WeightsResponse> {
  return getJSON<WeightsResponse>('/api/weights/current');
}

/**
 * Resolve the current status of a training job by lesson id.
 *
 * The training WebSocket only carries LIVE events and closes when the job ends,
 * so a tab that reconnects AFTER the job already finished (or failed) would sit
 * forever. This one-shot fetch reads the persisted lesson/job state so a
 * reconnecting tab can tell "still training" from "finished / blocked / errored".
 *
 * Returns null if the endpoint is unavailable (e.g. an older backend without it),
 * so callers can fall back to a plain WebSocket reconnect.
 */
export async function getTrainStatus(lessonId: number): Promise<TrainStatus | null> {
  try {
    return await getJSON<TrainStatus>(`/api/train/status/${lessonId}`);
  } catch {
    return null;
  }
}

/**
 * Opens a WebSocket to the training stream for a lesson.
 * Parses each JSON message into a TrainEvent and calls onEvent.
 * Returns a cleanup function that closes the socket.
 */
export function openTrainStream(
  lessonId: number,
  onEvent: (e: TrainEvent) => void,
  onClose?: () => void
): () => void {
  // WebSocket URL. Vercel's rewrites do NOT proxy WebSocket upgrades — a WS to
  // the Vercel origin arrives at the backend as a plain GET and 404s, leaving
  // the training card stuck. So in production we connect the WS DIRECTLY to the
  // backend (Railway supports WS natively) via NEXT_PUBLIC_BACKEND_WS_URL, e.g.
  // "wss://unrestrictedai-production.up.railway.app". Locally the var is unset
  // and we fall back to same-origin, which the Next dev server proxies fine.
  const base = process.env.NEXT_PUBLIC_BACKEND_WS_URL?.replace(/\/+$/, '');
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = base
    ? `${base}/api/train/stream/${lessonId}`
    : `${proto}//${location.host}/api/train/stream/${lessonId}`;
  const ws = new WebSocket(url);

  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data as string) as TrainEvent;
      onEvent(data);
    } catch {
      // Ignore malformed frames.
    }
  };

  if (onClose) {
    ws.onclose = () => onClose();
  }

  return () => {
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close();
    }
  };
}

/** Compact relative time, e.g. "just now", "2m ago", "3h ago", "5d ago". */
export function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '';
  const seconds = Math.floor((Date.now() - then) / 1000);

  if (seconds < 10) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;

  const weeks = Math.floor(days / 7);
  if (weeks < 5) return `${weeks}w ago`;

  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;

  const years = Math.floor(days / 365);
  return `${years}y ago`;
}
