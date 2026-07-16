import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('../views/LoginView.vue'),
    meta: { title: '登录' },
  },
  {
    path: '/',
    component: () => import('../components/layout/AppLayout.vue'),
    meta: { requiresAuth: true },
    children: [
      {
        path: '',
        name: 'Dashboard',
        component: () => import('../views/DashboardView.vue'),
        meta: { title: '工作台' },
      },
      {
        path: 'chat',
        name: 'Chat',
        component: () => import('../views/ChatView.vue'),
        meta: { title: '智能客服' },
      },
      {
        path: 'admin',
        name: 'Admin',
        component: () => import('../views/AdminView.vue'),
        meta: { title: '知识库管理', requiresMerchant: true },
      },
    ],
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// 路由守卫：未登录跳登录页；角色不匹配跳工作台
router.beforeEach((to, _from, next) => {
  const token = localStorage.getItem('access_token')

  if (to.meta.requiresAuth && !token) {
    next('/login')
    return
  }

  if (to.path === '/login' && token) {
    next('/')
    return
  }

  // 商户路由守卫：非商户角色无法访问知识库管理
  if (to.meta.requiresMerchant) {
    try {
      const payload = JSON.parse(atob(token!.split('.')[1]))
      const role = payload.role
      if (role !== 'admin' && role !== 'merchant') {
        next('/')
        return
      }
    } catch {
      next('/login')
      return
    }
  }

  next()
})

export default router
