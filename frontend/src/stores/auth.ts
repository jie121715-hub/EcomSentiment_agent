import { defineStore } from 'pinia'
import { ref } from 'vue'
import { authApi } from '../api/auth'

export interface UserInfo {
  user_id: string
  role: string
  merchant_id: string
  username: string
}

export const useAuthStore = defineStore('auth', () => {
  const token = ref<string | null>(localStorage.getItem('access_token'))
  const user = ref<UserInfo | null>(null)

  const isLoggedIn = () => !!token.value
  const isAdmin = () => user.value?.role === 'admin'
  const isMerchant = () => user.value?.role === 'admin' || user.value?.role === 'merchant'

  async function login(username: string, password: string) {
    const res = await authApi.login(username, password)
    token.value = res.access_token
    localStorage.setItem('access_token', res.access_token)
    await fetchMe()
    return res
  }

  async function fetchMe() {
    if (!token.value) return
    try {
      const me = await authApi.me()
      user.value = me as unknown as UserInfo
    } catch {
      logout()
    }
  }

  function logout() {
    token.value = null
    user.value = null
    localStorage.removeItem('access_token')
  }

  return { token, user, isLoggedIn, isAdmin, isMerchant, login, fetchMe, logout }
})
