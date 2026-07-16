import client from './client'

export interface ChatRequest {
  query: string
  user_id: string
  session_id: string
  shop_id?: string
  history: Array<{ question: string; answer: string }>
}

export interface ChatResponse {
  success: boolean
  message: {
    role: string
    content: string
    sentiment_detected: string
    intent_detected: string
  }
  processing_time_ms: number
}

export const chatApi = {
  /** 非流式对话 */
  send(req: ChatRequest): Promise<ChatResponse> {
    return client.post('/chat', req).then(r => r.data)
  },

  /** 流式对话 — 返回 fetch Response 用于 SSE 消费 */
  sendStream(req: ChatRequest): Promise<Response> {
    const token = localStorage.getItem('access_token')
    return fetch(`${import.meta.env.VITE_API_BASE_URL || '/api/v1'}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(req),
    })
  },
}
