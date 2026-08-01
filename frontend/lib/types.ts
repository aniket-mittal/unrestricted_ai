// Shared types for the DUM-E frontend.
// These mirror the FastAPI backend contract exactly. Do not add fields.

export interface ChatRequest {
  conversation_id?: number;
  message: string;
  user_id?: string;
  /**
   * Stable per-browser conversation id (UUID from chatStore). Chats live in the
   * browser now; this identifies the conversation for rate-limiting without an
   * account. Optional so the client stays compatible with the pre-client-history
   * backend (which ignores it).
   */
  client_id?: string;
  /**
   * Recent turns the client already holds. Once chat history lives in the
   * browser, the server no longer stores it, so the client sends the context it
   * wants the model to see. Ignored by the older backend (which reads history
   * from its own DB), used by the newer one.
   */
  history?: HistoryTurn[];
  /**
   * Concepts already taught + trained in THIS chat (from the client's persistent
   * lesson records). The server feeds these to the teaching detector as an
   * "already taught" note so a later recall question about a taught fact isn't
   * mis-detected as a new teach. Ignored by older backends.
   */
  taught_concepts?: string[];
}

/** A prior turn sent from the client so the server needn't store chat history. */
export interface HistoryTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface PromptResponsePair {
  prompt: string;
  response: string;
}

export interface ToolCallOut {
  concept: string;
  kind?: string; // fact | style | behavior — selects training knobs
  num_pairs: number;
  core_ratio?: number;
  pairs: PromptResponsePair[];
  summary: string;
}

export interface ChatResponse {
  conversation_id: number;
  reply: string;
  tool_call: ToolCallOut | null;
}

export interface LessonRequest {
  conversation_id?: number;
  /** Stable per-browser id used for the anti-runaway rate cap (no account). */
  client_id?: string;
  concept: string;
  kind?: string; // fact | style | behavior — selects training knobs
  num_pairs: number;
  core_ratio?: number;
  pairs: PromptResponsePair[];
  summary: string;
}

/** Response of GET /api/train/status/{lesson_id} — resolves a job after the WS closed. */
export interface TrainStatus {
  lesson_id: number;
  /** queued | training | done | blocked | error (mirrors the lessons table). */
  status: string;
  version: string | null;
  final_loss: number | null;
  blocked_reason: string | null;
}

export interface LessonResponse {
  lesson_id: number;
  status: string;
  num_pairs: number;
  blocked_reason: string | null;
}

export interface FeedItem {
  id: number;
  lesson_id: number | null;
  summary: string;
  created_at: string;
}

export interface WeightsResponse {
  version_id: number | null;
  kind: string | null;
  path: string | null;
  created_at: string | null;
}

// Training WebSocket event union.
export type TrainEvent =
  // Emitted while the worker fans out the Gemini augmentation (the real
  // sample-generation, off the request path). The card stays in "generating".
  | { type: 'augment'; lesson_id: number; status: string }
  | { type: 'progress'; lesson_id: number; step: number; total_steps: number; loss: number }
  | { type: 'done'; lesson_id: number; version: string; path: string; kind: string; train_s: number; final_loss?: number }
  | { type: 'retry'; lesson_id: number; attempt: number; error: string }
  | { type: 'error'; lesson_id: number; error: string };

// Local UI-only message type (not a backend shape).
// role 'event' is a persistent inline record of a completed lesson (the
// "generated N samples -> learned X" chip that stays in the thread).
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'event';
  content: string;
  toolCall?: ToolCallOut;
  event?: {
    numPairs: number;
    summary: string;
    version: string;
  };
}
