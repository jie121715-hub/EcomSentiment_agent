<template>
  <div class="sidebar">
    <div class="logo">
      <el-icon :size="24"><ChatDotRound /></el-icon>
      <span class="logo-text">云答</span>
    </div>
    <el-menu
      :default-active="activeMenu"
      router
      background-color="#1d1e2c"
      text-color="#a3a6b4"
      active-text-color="#409eff"
      class="sidebar-menu"
    >
      <el-menu-item index="/">
        <el-icon><HomeFilled /></el-icon>
        <span>工作台</span>
      </el-menu-item>

      <!-- 商户：知识库管理 -->
      <template v-if="auth.isMerchant()">
        <el-menu-item index="/admin">
          <el-icon><FolderOpened /></el-icon>
          <span>知识库管理</span>
        </el-menu-item>
      </template>

      <!-- 普通用户：智能客服 -->
      <template v-else>
        <el-menu-item index="/chat">
          <el-icon><ChatLineSquare /></el-icon>
          <span>智能客服</span>
        </el-menu-item>
      </template>
    </el-menu>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from '../../stores/auth'

const route = useRoute()
const auth = useAuthStore()
const activeMenu = computed(() => route.path)
</script>

<style scoped>
.sidebar {
  height: 100%;
  display: flex;
  flex-direction: column;
}
.logo {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
  color: #409eff;
}
.logo-text {
  font-size: 18px;
  font-weight: 700;
}
.sidebar-menu {
  border-right: none;
}
</style>
