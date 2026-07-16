<template>
  <div class="admin-view">
    <h2>📝 知识库管理</h2>

    <!-- 数据源切换 -->
    <el-radio-group v-model="dataSource" style="margin-bottom:12px">
      <el-radio-button value="mysql">MySQL 知识库</el-radio-button>
      <el-radio-button value="milvus">Milvus eco_rag</el-radio-button>
    </el-radio-group>

    <!-- 操作栏（仅 MySQL 模式显示） -->
    <el-card class="action-card" v-if="dataSource === 'mysql'">
      <el-space wrap>
        <el-select v-model="form.category" style="width:120px" placeholder="选择分类">
          <el-option value="product" label="商品" />
          <el-option value="policy" label="政策" />
          <el-option value="faq" label="FAQ" />
        </el-select>
        <el-upload
          :show-file-list="false"
          :before-upload="handleUpload"
          accept=".pdf,.docx,.md,.markdown,.txt"
          :disabled="uploading"
        >
          <el-button type="primary" :icon="Upload" :loading="uploading">
            {{ uploading ? '上传中...' : '上传文件' }}
          </el-button>
        </el-upload>
        <el-button type="success" @click="showAddDialog = true" :icon="Plus">
          手动添加
        </el-button>
        <el-button @click="handleSync" :icon="Refresh" :loading="syncing">
          同步向量库
        </el-button>
        <el-button type="warning" @click="showUserDialog = true" :icon="UserFilled">
          创建用户
        </el-button>
      </el-space>
    </el-card>

    <!-- 知识列表（MySQL） -->
    <el-card v-if="dataSource === 'mysql'">
      <template #header>
        <div class="list-header">
          <span>知识列表（共 {{ total }} 条）</span>
          <el-input
            v-model="searchCategory"
            placeholder="按分类筛选"
            clearable
            style="width: 200px"
            @change="loadList"
          />
        </div>
      </template>
      <el-table :data="list" stripe v-loading="loading" size="small">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="content" label="内容" show-overflow-tooltip min-width="300" />
        <el-table-column prop="category" label="分类" width="100" />
        <el-table-column prop="merchant_id" label="商户" width="100" />
        <el-table-column prop="created_at" label="创建时间" width="160" />
        <el-table-column label="操作" width="80">
          <template #default="{ row }">
            <el-button type="danger" size="small" @click="handleDelete(row.id)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class="pagination-wrap" v-if="total > 0">
        <el-pagination
          v-model:current-page="currentPage"
          :page-size="pageSize"
          :total="total"
          layout="total, prev, pager, next, jumper"
          @current-change="loadList"
        />
      </div>
    </el-card>

    <!-- 知识列表（Milvus eco_rag） -->
    <el-card v-if="dataSource === 'milvus'">
      <template #header>
        <div class="list-header">
          <span>eco_rag · {{ ragCollection === 'policies' ? '政策' : '商品' }}（共 {{ ragTotal }} 条，shop_id={{ ragShopId }}）</span>
          <el-radio-group v-model="ragCollection" size="small" @change="loadRagList">
            <el-radio-button value="policies">政策</el-radio-button>
            <el-radio-button value="products">商品</el-radio-button>
          </el-radio-group>
        </div>
      </template>
      <el-table :data="ragItems" stripe v-loading="ragLoading" size="small">
        <el-table-column prop="doc_id" label="文档ID" width="200" show-overflow-tooltip />
        <el-table-column prop="content" label="内容" show-overflow-tooltip min-width="300" />
        <el-table-column prop="category" label="分类" width="100" />
      </el-table>
    </el-card>

    <!-- 添加知识对话框 -->
    <el-dialog v-model="showAddDialog" title="添加知识" width="500px">
      <el-form :model="form">
        <el-form-item label="内容">
          <el-input v-model="form.content" type="textarea" :rows="4" />
        </el-form-item>
        <el-form-item label="分类">
          <el-select v-model="form.category">
            <el-option value="product" label="商品" />
            <el-option value="policy" label="政策" />
            <el-option value="faq" label="FAQ" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddDialog = false">取消</el-button>
        <el-button type="primary" @click="handleAdd" :loading="adding">确认</el-button>
      </template>
    </el-dialog>

    <!-- 创建用户对话框 -->
    <el-dialog v-model="showUserDialog" title="创建用户" width="420px">
      <el-form :model="userForm">
        <el-form-item label="用户名">
          <el-input v-model="userForm.username" placeholder="请输入用户名" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="userForm.password" type="password" show-password placeholder="请输入密码" />
        </el-form-item>
        <el-form-item label="角色">
          <el-select v-model="userForm.role">
            <el-option value="customer" label="用户（customer）" />
            <el-option value="merchant" label="商家（merchant）" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showUserDialog = false">取消</el-button>
        <el-button type="primary" @click="handleCreateUser" :loading="creatingUser">确认创建</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, reactive, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Upload, Plus, Refresh, UserFilled } from '@element-plus/icons-vue'
import { adminApi, type KnowledgeItem } from '../api/admin'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()

const list = ref<KnowledgeItem[]>([])
const total = ref(0)
const loading = ref(false)
const syncing = ref(false)
const adding = ref(false)
const showAddDialog = ref(false)
const searchCategory = ref('')

const dataSource = ref('mysql')
const currentPage = ref(1)
const pageSize = 10

// eco_rag
const ragItems = ref<any[]>([])
const ragTotal = ref(0)
const ragShopId = ref('')
const ragLoading = ref(false)
const ragCollection = ref('policies')

const form = reactive({ content: '', category: 'product' })

// 用户创建
const showUserDialog = ref(false)
const creatingUser = ref(false)
const userForm = reactive({ username: '', password: '', role: 'customer' })

onMounted(() => {
  loadList()
  // 默认不加载 Milvus，切过去再加载
})

async function loadRagList() {
  ragLoading.value = true
  try {
    const data = await adminApi.listRagKnowledge(ragCollection.value)
    ragItems.value = data.items || []
    ragTotal.value = data.total || 0
    ragShopId.value = data.shop_id || ''
  } catch {
    ragItems.value = []
  } finally {
    ragLoading.value = false
  }
}

// 监听切换 dataSource
watch(dataSource, (val) => {
  if (val === 'milvus') loadRagList()
})

async function loadList() {
  loading.value = true
  try {
    const offset = (currentPage.value - 1) * pageSize
    const res = await adminApi.listKnowledge({
      category: searchCategory.value || undefined,
      offset,
      limit: pageSize,
    })
    list.value = res.items
    total.value = res.total
  } finally {
    loading.value = false
  }
}

async function handleAdd() {
  if (!form.content.trim()) return
  adding.value = true
  try {
    // 1. 存 MySQL
    const mid = auth.user?.merchant_id || 'default'
    await adminApi.addKnowledge({ content: form.content, category: form.category, merchant_id: mid })
    // 2. 存 Milvus eco_rag（带shop_id隔离）
    try {
      const r = await adminApi.uploadText(form.content, form.category)
      ElMessage.success(`添加成功, Milvus: ${r.chunks.parents}父块+${r.chunks.children}子块`)
    } catch {
      ElMessage.warning('已保存到数据库，但向量写入失败')
    }
    showAddDialog.value = false
    form.content = ''
    currentPage.value = 1
    loadList()
  } finally {
    adding.value = false
  }
}

async function handleDelete(id: number) {
  try {
    await ElMessageBox.confirm('确认删除？', '提示', { type: 'warning' })
    await adminApi.deleteKnowledge(id)
    ElMessage.success('已删除')
    loadList()
  } catch {}
}

async function handleSync() {
  syncing.value = true
  try {
    const res = await adminApi.syncKnowledge()
    ElMessage.success(`同步完成：${res.synced} 条知识，${res.chunks} 个向量块`)
  } finally {
    syncing.value = false
  }
}

const uploading = ref(false)

async function handleUpload(file: File) {
  uploading.value = true
  try {
    const res = await adminApi.uploadFile(file, form.category)
    if (res.success) {
      ElMessage.success(`上传成功：${res.chunks?.children || 0} 个子块已入库，${res.eco_rag_ok ? '向量已写入' : ''}`)
      currentPage.value = 1
      loadList()
      if (dataSource.value === 'milvus') loadRagList()
    } else {
      ElMessage.error(res.error || '上传失败')
    }
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '上传超时，请重试')
  } finally {
    uploading.value = false
  }
  return false
}

async function handleCreateUser() {
  if (!userForm.username || !userForm.password) {
    ElMessage.warning('请填写用户名和密码')
    return
  }
  creatingUser.value = true
  try {
    await adminApi.createUser(userForm.username, userForm.password, userForm.role)
    ElMessage.success(`用户 ${userForm.username}（${userForm.role}）创建成功`)
    showUserDialog.value = false
    userForm.username = ''
    userForm.password = ''
    userForm.role = 'customer'
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '创建失败')
  } finally {
    creatingUser.value = false
  }
}
</script>

<style scoped>
.admin-view { max-width: 1100px; margin: 0 auto; }
.admin-view h2 { margin-bottom: 20px; }
.action-card { margin-bottom: 20px; }
.list-header { display: flex; justify-content: space-between; align-items: center; }
.pagination-wrap {
  display: flex;
  justify-content: center;
  margin-top: 16px;
}
</style>
