// Shared types for the DUM-E frontend.
// These mirror the FastAPI backend contract exactly. Do not add fields.

export interface ChatRequest {
  conversation_id?: number;
  message: string;
  user_id?: string;
}

export interface PromptResponsePair {
  prompt: string;
  response: string;
}

export interface ToolCallOut {
  concept: string;
  num_pairs: number;
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
  concept: string;
  num_pairs: number;
  pairs: PromptResponsePair[];
  summary: string;
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
  | { type: 'progress'; lesson_id: number; step: number; total_steps: number; loss: number }
  | { type: 'done'; lesson_id: number; version: string; path: string; kind: string; train_s: number; final_loss?: number }
  | { type: 'retry'; lesson_id: number; attempt: number; error: string }
  | { type: 'error'; lesson_id: number; error: string };

// Local UI-only message type (not a backend shape).
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  toolCall?: ToolCallOut;
}
