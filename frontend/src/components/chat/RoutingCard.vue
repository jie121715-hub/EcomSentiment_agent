<template>
  <el-card class="routing-card" shadow="never" v-if="data">
    <template #header>
      <span class="card-title">🧠 路由决策</span>
    </template>
    <div class="info-row">
      <span class="label">目标Agent：</span>
      <el-tag size="small" type="primary">{{ agentLabel }}</el-tag>
    </div>
    <div class="info-row" v-if="data.urgency && data.urgency !== 'normal'">
      <span class="label">紧急度：</span>
      <el-tag size="small" :type="data.urgency === 'critical' ? 'danger' : 'warning'">
        {{ data.urgency === 'critical' ? '⚠️ 紧急' : '⚡ 优先' }}
      </el-tag>
    </div>
    <div class="info-row" v-if="data.strategy">
      <span class="label">策略：</span>
      <span class="value">{{ data.strategy }}</span>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ data: any }>()

const agentLabel = computed(() => {
  const map: Record<string, string> = {
    knowledge_qa: '📚 知识应答Agent',
    business: '📦 业务处理Agent',
    escalate: '📞 人工客服',
  }
  return map[props.data?.target_agent] || props.data?.target_agent || '—'
})
</script>

<style scoped>
.routing-card {
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
}
.label {
  color: #909399;
  min-width: 72px;
}
.value {
  color: #606266;
}
</style>
