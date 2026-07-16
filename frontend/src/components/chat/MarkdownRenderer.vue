<template>
  <div class="markdown-body" v-html="renderedHtml" />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'

const md = new MarkdownIt({
  breaks: true,
  linkify: true,
  html: true,
})

const props = defineProps<{ content: string }>()
const renderedHtml = computed(() => md.render(props.content))
</script>

<style>
.markdown-body {
  line-height: 1.7;
  word-break: break-word;
}
.markdown-body p {
  margin: 0 0 8px;
}
.markdown-body pre {
  background: #f5f7fa;
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
}
.markdown-body code {
  background: #f0f2f5;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 13px;
}
.markdown-body ul, .markdown-body ol {
  padding-left: 20px;
}
</style>
