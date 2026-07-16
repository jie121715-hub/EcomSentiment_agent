import { ref } from 'vue'
import { chatApi } from '../api/chat'

export interface SSEMessage {
  event: string
  data: string
}

export interface PerceptionData {
  sentiment: string
  sentiment_label: string
  intent: string
  entities: Array<{ type: string; value: string }>
  intent_confidence: number
}

export interface RouteData {
  target_agent: string
  urgency: string
  strategy: string
  escalate_to_human: boolean
}

export function useSSEChat() {
  const isStreaming = ref(false)
  const streamContent = ref('')
  const perception = ref<PerceptionData | null>(null)
  const route = ref<RouteData | null>(null)

  async function sendMessage(query: string, userId: string, sessionId: string, shopId: string = '') {
    isStreaming.value = true
    streamContent.value = ''
    perception.value = null
    route.value = null

    try {
      const response = await chatApi.sendStream({
        query,
        user_id: userId,
        session_id: sessionId,
        shop_id: shopId,
        history: [],
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let currentEvent = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            const data = line.slice(6)
            handleEvent(currentEvent, data)
          }
        }
      }
    } catch (e: any) {
      streamContent.value += `\n\n❌ 连接错误: ${e.message}`
    } finally {
      isStreaming.value = false
    }
  }

  function handleEvent(event: string, data: string) {
    switch (event) {
      case 'perception':
        try {
          perception.value = JSON.parse(data)
        } catch {}
        break
      case 'route':
        try {
          route.value = JSON.parse(data)
        } catch {}
        break
      case 'token':
        streamContent.value += data
        break
      case 'done':
        // 完成
        break
      case 'error':
        streamContent.value += `\n\n❌ ${data}`
        break
    }
  }

  return { isStreaming, streamContent, perception, route, sendMessage }
}
