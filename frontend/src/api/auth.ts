import client from './client'

export const authApi = {
  login(username: string, password: string) {
    return client.post('/auth/login', { username, password }).then(r => r.data)
  },
  register(username: string, password: string, phone?: string, role?: string, merchant_id?: string) {
    return client.post('/auth/register', {
      username, password,
      phone: phone || null,
      role: role || null,
      merchant_id: merchant_id || null,
    }).then(r => r.data)
  },
  me() {
    return client.get('/auth/me').then(r => r.data)
  },
}
