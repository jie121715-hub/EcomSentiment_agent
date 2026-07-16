<template>
  <el-card class="perception-card" shadow="never" v-if="data">
    <template #header>
      <span class="card-title">🎯 感知结果</span>
    </template>
    <div class="info-row">
      <span class="label">情感：</span>
      <el-tag size="small" :type="sentimentType">{{ data.sentiment_label || data.sentiment }}</el-tag>
    </div>
    <div class="info-row">
      <span class="label">意图：</span>
      <el-tag size="small" type="success">{{ intentLabel }}</el-tag>
    </div>
    <div class="info-row" v-if="data.entities?.length">
      <span class="label">实体：</span>
      <el-tag v-for="(e, i) in data.entities" :key="i" size="small" type="info" class="entity-tag">
        {{ e.type }}: {{ e.value }}
      </el-tag>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ data: any }>()

const sentimentType = computed(() => {
  const s = props.data?.sentiment
  if (s === 'positive' || s === 'happy' || s === 'grateful') return 'success'
  if (s === 'negative' || s === 'angry' || s === 'disappointed') return 'danger'
  return 'warning'
})

const intentLabel = computed(() => {
  const map: Record<string, string> = {
    knowledge_qa: '知识问答',
    business: '业务处理',
    escalate: '转人工',
  }
  return map[props.data?.intent] || props.data?.intent || '—'
})
</script>

<style scoped>
.perception-card {
  margin-bottom: 12px;
  max-width: 400px;
  font-size: 13px;
}
.card-title {
  font-weight: 600;
  font-size: 13px;
}
.info-row {
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.label {
  color: #909399;
  min-width: 48px;
}
.entity-tag {
  margin-right: 4px;
}
</style>
