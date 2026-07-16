<template>
  <div :class="['chat-bubble', role]">
    <div class="bubble-header" v-if="role === 'assistant' && (sentiment || intent)">
      <el-tag v-if="sentiment" size="small" type="warning" class="tag">🎯 {{ sentiment }}</el-tag>
      <el-tag v-if="intent" size="small" type="success" class="tag">📋 {{ intent }}</el-tag>
    </div>
    <div class="bubble-content">
      <MarkdownRenderer v-if="role === 'assistant'" :content="content" />
      <template v-else>{{ content }}</template>
    </div>
  </div>
</template>

<script setup lang="ts">
import MarkdownRenderer from './MarkdownRenderer.vue'

defineProps<{
  content: string
  role: 'user' | 'assistant'
  sentiment?: string
  intent?: string
}>()
</script>

<style scoped>
.chat-bubble {
  margin-bottom: 16px;
  max-width: 80%;
}
.chat-bubble.user {
  margin-left: auto;
}
.chat-bubble.user .bubble-content {
  background: #409eff;
  color: #fff;
  border-radius: 12px 12px 0 12px;
  padding: 10px 16px;
}
.chat-bubble.assistant .bubble-content {
  background: #fff;
  border: 1px solid #e4e7ed;
  border-radius: 12px 12px 12px 0;
  padding: 12px 16px;
}
.bubble-header {
  margin-bottom: 4px;
  display: flex;
  gap: 6px;
}
.tag {
  font-size: 11px;
}
</style>
