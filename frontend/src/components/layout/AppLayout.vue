<template>
  <el-container class="app-layout">
    <el-aside width="220px">
      <Sidebar />
    </el-aside>
    <el-container>
      <el-header height="56px">
        <Header />
      </el-header>
      <el-main>
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import Sidebar from './Sidebar.vue'
import Header from './Header.vue'
</script>

<style scoped>
.app-layout {
  height: 100vh;
}
.el-aside {
  background-color: #1d1e2c;
  color: #fff;
  overflow-x: hidden;
}
.el-header {
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
  padding: 0 20px;
}
.el-main {
  background: #f5f7fa;
  padding: 20px;
  overflow-y: auto;
}
</style>
