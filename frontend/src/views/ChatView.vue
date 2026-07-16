<template>
  <div class="chat-view">
    <div class="chat-messages" ref="msgContainer">
      <div v-if="messages.length === 0" class="welcome">
        <h2>👋 您好，我是智能客服助手</h2>
        <p>可以帮您查询商品、物流、订单，或回答店铺政策相关问题</p>
        <div class="quick-questions">
          <el-tag
            v-for="q in quickQuestions"
            :key="q"
            class="quick-tag"
            @click="handleQuick(q)"
          >
            {{ q }}
          </el-tag>
        </div>
      </div>
      <template v-for="(msg, i) in messages" :key="i">
        <PerceptionCard v-if="msg.perception" :data="msg.perception" />
        <RoutingCard v-if="msg.route" :data="msg.route" />
        <ChatBubble
          :content="msg.content"
          :role="msg.role"
          :sentiment="msg.sentiment"
          :intent="msg.intent"
        />
      </template>
      <div v-if="sse.isStreaming.value && sse.streamContent.value" class="streaming-bubble">
        <MarkdownRenderer :content="sse.streamContent.value" />
        <span class="cursor">▊</span>
      </div>
    </div>
    <ChatInput :disabled="sse.isStreaming.value" @send="handleSend" />
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch, onMounted } from 'vue'
import { useSSEChat } from '../composables/useSSEChat'
import { useAuthStore } from '../stores/auth'
import ChatBubble from '../components/chat/ChatBubble.vue'
import ChatInput from '../components/chat/ChatInput.vue'
import MarkdownRenderer from '../components/chat/MarkdownRenderer.vue'
import PerceptionCard from '../components/chat/PerceptionCard.vue'
import RoutingCard from '../components/chat/RoutingCard.vue'

interface Message {
  content: string
  role: 'user' | 'assistant'
  sentiment?: string
  intent?: string
  perception?: any
  route?: any
}

const sse = useSSEChat()
const auth = useAuthStore()
const msgContainer = ref<HTMLElement>()
const messages = ref<Message[]>([])

const quickQuestions = [
  '这件衣服是什么材质的？',
  '我的订单到哪了？',
  '如何申请退款？',
  '店铺支持7天无理由退货吗？',
  '有没有推荐的新品？',
]

onMounted(async () => {
  if (!auth.user) await auth.fetchMe()
})

function scrollToBottom() {
  nextTick(() => {
    const el = msgContainer.value
    if (el) el.scrollTop = el.scrollHeight
  })
}

async function handleSend(text: string) {
  // 添加用户消息
  messages.value.push({ content: text, role: 'user' })
  scrollToBottom()

  // 发送请求
  const userId = auth.user?.username || 'web_user'
  const shopId = auth.user?.merchant_id || ''
  await sse.sendMessage(text, userId, userId, shopId)

  // 添加助手消息（含感知和路由信息）
  if (sse.streamContent.value || sse.perception.value || sse.route.value) {
    messages.value.push({
      content: sse.streamContent.value || '抱歉，未能获取到回复',
      role: 'assistant',
      sentiment: sse.perception.value?.sentiment_label,
      intent: sse.perception.value?.intent,
      perception: sse.perception.value ? { ...sse.perception.value } : undefined,
      route: sse.route.value ? { ...sse.route.value } : undefined,
    })
  }
  scrollToBottom()
}

function handleQuick(q: string) {
  handleSend(q)
}

watch(() => sse.streamContent.value, scrollToBottom)
</script>

<style scoped>
.chat-view {
  height: calc(100vh - 56px - 40px);
  display: flex;
  flex-direction: column;
  max-width: 900px;
  margin: 0 auto;
}
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}
.welcome {
  text-align: center;
  padding: 60px 20px;
  color: #606266;
}
.welcome h2 {
  margin-bottom: 8px;
}
.quick-questions {
  margin-top: 20px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
}
.quick-tag {
  cursor: pointer;
}
.quick-tag:hover {
  background: #409eff;
  color: #fff;
}
.streaming-bubble {
  background: #fff;
  border: 1px solid #e4e7ed;
  border-radius: 12px 12px 12px 0;
  padding: 12px 16px;
  max-width: 80%;
  position: relative;
}
.cursor {
  animation: blink 1s infinite;
  color: #409eff;
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
</style>
