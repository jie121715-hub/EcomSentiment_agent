<template>
  <div class="chat-input">
    <el-input
      v-model="text"
      type="textarea"
      :rows="2"
      placeholder="输入您的问题..."
      @keydown.enter.exact.prevent="send"
      :disabled="disabled"
    />
    <el-button
      type="primary"
      :icon="Promotion"
      :disabled="!text.trim() || disabled"
      @click="send"
      :loading="disabled"
    >
      发送
    </el-button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Promotion } from '@element-plus/icons-vue'

const props = defineProps<{ disabled?: boolean }>()
const emit = defineEmits<{ send: [text: string] }>()

const text = ref('')

function send() {
  const t = text.value.trim()
  if (!t || props.disabled) return
  emit('send', t)
  text.value = ''
}
</script>

<style scoped>
.chat-input {
  display: flex;
  gap: 12px;
  align-items: flex-end;
  padding: 16px;
  background: #fff;
  border-top: 1px solid #e4e7ed;
}
.chat-input .el-button {
  height: 40px;
}
</style>
