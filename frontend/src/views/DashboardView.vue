<template>
  <div class="dashboard">
    <h2>📊 工作台</h2>
    <el-row :gutter="20" class="stats-row">
      <el-col :span="8">
        <el-card shadow="hover">
          <div class="stat-card">
            <el-icon :size="32" color="#409eff"><ChatLineSquare /></el-icon>
            <div class="stat-info">
              <div class="stat-num">{{ stats.todayChats }}</div>
              <div class="stat-label">今日对话</div>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="8">
        <el-card shadow="hover">
          <div class="stat-card">
            <el-icon :size="32" color="#e6a23c"><Tickets /></el-icon>
            <div class="stat-info">
              <div class="stat-num">{{ stats.tickets }}</div>
              <div class="stat-label">待处理工单</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover">
          <div class="stat-card">
            <el-icon :size="32" color="#f56c6c"><Service /></el-icon>
            <div class="stat-info">
              <div class="stat-num">{{ stats.avgTime }}ms</div>
              <div class="stat-label">平均响应</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-card class="section-card">
      <template #header><h3>🚀 快速操作</h3></template>
      <el-space wrap>
        <el-button type="primary" @click="$router.push('/chat')">
          <el-icon><ChatDotRound /></el-icon> 开始对话
        </el-button>

        <el-button @click="handleCheckHealth">
          <el-icon><Monitor /></el-icon> 系统健康检查
        </el-button>
      </el-space>
    </el-card>

    <el-card class="section-card">
      <template #header><h3>🤖 Agent 体系状态</h3></template>
      <el-table :data="agentList" size="small">
        <el-table-column prop="name" label="Agent" />
        <el-table-column prop="role" label="职责" />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="row.status === '就绪' ? 'success' : 'warning'" size="small">
              {{ row.status }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { reactive } from 'vue'
import { ElMessage } from 'element-plus'
import client from '../api/client'

const stats = reactive({
  todayChats: '—',
  tickets: '—',
  avgTime: '—',
})

const agentList = [
  { name: '🎯 感知Agent', role: '情感分析 + 意图识别 + 实体抽取', status: '就绪' },
  { name: '🧠 路由Agent', role: '三维决策：置信度/紧急度/分发', status: '就绪' },
  { name: '📚 知识应答Agent', role: 'Redis→BM25→RAG 三层检索', status: '就绪' },
  { name: '📦 业务Agent', role: '查物流/订单/库存+改地址/退款', status: '就绪' },

]

async function handleCheckHealth() {
  try {
    const res = await client.get('/health')
    ElMessage.success(`服务正常 — v${res.data.version}`)
  } catch {
    ElMessage.error('服务异常')
  }
}
</script>

<style scoped>
.dashboard { max-width: 1100px; margin: 0 auto; }
.dashboard h2 { margin-bottom: 20px; }
.stats-row { margin-bottom: 20px; }
.stat-card { display: flex; align-items: center; gap: 16px; }
.stat-num { font-size: 28px; font-weight: 700; color: #303133; }
.stat-label { font-size: 13px; color: #909399; }
.section-card { margin-bottom: 20px; }
</style>
