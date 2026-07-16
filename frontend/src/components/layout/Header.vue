<template>
  <div class="header">
    <span class="title">🛒 云答智能客服系统</span>
    <div class="header-right">
      <el-tag v-if="auth.user" size="small" type="info">
        {{ auth.user.role === 'admin' ? '管理员' : auth.user.role === 'merchant' ? '商家' : '用户' }}
      </el-tag>
      <el-button type="danger" text @click="handleLogout">退出</el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useAuthStore } from '../../stores/auth'

const auth = useAuthStore()
const router = useRouter()

function handleLogout() {
  auth.logout()
  router.push('/login')
}
</script>

<style scoped>
.header {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.title {
  font-size: 16px;
  font-weight: 600;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}
</style>
