// Thin, dependency-free API client for the DUM-E frontend.
// All requests are same-origin; the Next.js rewrite proxies /api to FastAPI.

import type {
  ChatRequest,
  ChatResponse,
  FeedItem,
  LessonRequest,
  LessonResponse,
  TrainEvent,
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

export function postLesson(req: LessonRequest): Promise<LessonResponse> {
  return postJSON<LessonResponse>('/api/lessons', req);
}

export function getLearned(limit = 50): Promise<FeedItem[]> {
  return getJSON<FeedItem[]>(`/api/learned?limit=${limit}`);
}

export function getCurrentWeights(): Promise<WeightsResponse> {
  return getJSON<WeightsResponse>('/api/weights/current');
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
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/api/train/stream/${lessonId}`;
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
