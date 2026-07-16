import client from './client'

export interface KnowledgeItem {
  id?: number
  content: string
  category: string
  merchant_id: string
  source?: string
  created_at?: string
}

export const adminApi = {
  /** 知识库列表（分页） */
  listKnowledge(params?: { category?: string; merchant_id?: string; offset?: number; limit?: number }) {
    return client.get('/admin/knowledge', { params }).then(r => r.data)
  },

  /** 添加知识 */
  addKnowledge(item: KnowledgeItem) {
    return client.post('/admin/knowledge', item).then(r => r.data)
  },

  /** 批量添加 */
  batchAddKnowledge(items: KnowledgeItem[]) {
    return client.post('/admin/knowledge/batch', items).then(r => r.data)
  },

  /** 删除知识 */
  deleteKnowledge(id: number) {
    return client.delete(`/admin/knowledge/${id}`).then(r => r.data)
  },

  /** eco_rag 知识列表 */
  listRagKnowledge(collection?: string) {
    return client.get('/admin/knowledge/rag-list', { params: { collection: collection || 'policies' } }).then(r => r.data)
  },

  /** 上传文本到 eco_rag（新 Milvus），超时 2 分钟 */
  uploadText(content: string, category: string) {
    return client.post('/admin/knowledge/upload-text', { content, category }, { timeout: 120000 }).then(r => r.data)
  },

  /** 同步向量库 */
  syncKnowledge() {
    return client.post('/admin/knowledge/sync').then(r => r.data)
  },

  /** 上传 PDF/DOCX（JWT 鉴权，自动带 shop_id） */
  uploadFile(file: File, category: string) {
    const form = new FormData()
    form.append('file', file)
    form.append('category', category)
    return client.post('/admin/knowledge/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000,
    }).then(r => r.data)
  },

  /** 创建用户（仅管理员/商户） */
  createUser(username: string, password: string, role: string) {
    return client.post('/auth/users', { username, password }, { params: { role } }).then(r => r.data)
  },
}
