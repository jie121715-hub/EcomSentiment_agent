<template>
  <div class="login-page">
    <!-- ═══ 背景层 ═══ -->
    <div class="bg-orb bg-orb-1"></div>
    <div class="bg-orb bg-orb-2"></div>
    <div class="bg-orb bg-orb-3"></div>
    <div class="grid-bg"></div>
    <canvas ref="canvasRef"></canvas>

    <!-- ═══ 主容器 ═══ -->
    <div class="main-container">
      <!-- ── 左侧：品牌展示 ── -->
      <div class="brand-panel">
        <div class="brand-badge"><span class="dot"></span> AI-Powered Ecommerce</div>
        <h1 class="brand-title">智能客服<span class="highlight">·</span>新体验</h1>
        <p class="brand-desc">云答智能客服系统，基于大语言模型精准理解用户意图，实时情感感知，极速响应每一次对话。</p>
        <div class="feature-cards">
          <div class="feature-card">
            <div class="fc-icon">🧠</div>
            <div class="fc-text"><h4>多模型意图识别</h4><p>DeepSeek 驱动，5路精准分发</p></div>
          </div>
          <div class="feature-card">
            <div class="fc-icon">💬</div>
            <div class="fc-text"><h4>情感实时感知</h4><p>7 维度情绪分析，暖心交互</p></div>
          </div>
          <div class="feature-card">
            <div class="fc-icon">📚</div>
            <div class="fc-text"><h4>RAG 知识增强</h4><p>混合检索 + 精排，秒级精准回答</p></div>
          </div>
        </div>
        <div class="stats-row">
          <div class="stat-item"><div class="stat-num">99.7<span class="stat-unit">%</span></div><div class="stat-label">意图识别准确率</div></div>
          <div class="stat-item"><div class="stat-num">&lt;1.2<span class="stat-unit">s</span></div><div class="stat-label">平均响应时间</div></div>
          <div class="stat-item"><div class="stat-num">50<span class="stat-unit">万+</span></div><div class="stat-label">服务用户数</div></div>
        </div>
      </div>

      <!-- ── 右侧：认证卡片 ── -->
      <div class="auth-panel">
        <div class="auth-card">
          <!-- 头部 -->
          <div class="auth-card-header">
            <span class="auth-card-title">云答 · 客服</span>
            <button class="enterprise-link" v-if="currentMode === 'login'" @click="switchMode('enterprise')">
              企业账号登录
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
            </button>
          </div>

          <!-- 返回链接 -->
          <button class="back-link" v-if="currentMode !== 'login'" @click="switchMode('login')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
            返回用户登录
          </button>

          <!-- ═══ 用户登录 Tab ═══ -->
          <div v-if="currentMode === 'login'" class="login-tabs">
            <button class="login-tab" :class="{ active: loginTab === 'password' }" @click="switchLoginTab('password')">账号密码</button>
            <button class="login-tab" :class="{ active: loginTab === 'sms' }" @click="switchLoginTab('sms')">手机验证码</button>
            <div class="tab-slider" :class="{ right: loginTab === 'sms' }"></div>
          </div>

          <!-- ═══ 账号密码登录 ═══ -->
          <div v-if="currentMode === 'login' && loginTab === 'password'" class="form-section active">
            <div class="input-group">
              <label>账号 / 手机号</label>
              <div class="input-wrapper">
                <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="8" r="4"/><path d="M6 21v-2a4 4 0 014-4h4a4 4 0 014 4v2"/></svg></span>
                <input type="text" v-model="loginForm.username" placeholder="请输入账号或手机号" autocomplete="username" @input="updateCaptchaState">
              </div>
              <div class="field-error" :class="{ show: errors.loginUsername }">{{ errors.loginUsername }}</div>
            </div>
            <div class="input-group">
              <label>密码</label>
              <div class="input-wrapper">
                <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg></span>
                <input :type="showLoginPw ? 'text' : 'password'" v-model="loginForm.password" placeholder="请输入密码" autocomplete="current-password" @input="updateCaptchaState" @keyup.enter="handleLogin">
                <button class="pw-toggle" @click="showLoginPw = !showLoginPw">{{ showLoginPw ? '🙈' : '👁' }}</button>
              </div>
              <div class="field-error" :class="{ show: errors.loginPassword }">{{ errors.loginPassword }}</div>
            </div>
            <div class="form-options">
              <label class="remember-me">
                <input type="checkbox" v-model="rememberMe">
                <span class="cb-box"></span> 记住我
              </label>
              <span class="forgot-link" @click="showToast('info', '请联系管理员重置密码')">忘记密码？</span>
            </div>
          </div>

          <!-- ═══ 手机验证码登录 ═══ -->
          <div v-if="currentMode === 'login' && loginTab === 'sms'" class="form-section active">
            <div class="input-group">
              <label>手机号</label>
              <div class="input-wrapper">
                <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="5" y="2" width="14" height="20" rx="2"/><line x1="12" y1="18" x2="12" y2="18.01"/></svg></span>
                <input type="tel" v-model="smsForm.phone" placeholder="请输入手机号" maxlength="11" @input="updateCaptchaState">
              </div>
              <div class="field-error" :class="{ show: errors.smsPhone }">{{ errors.smsPhone }}</div>
            </div>
            <div class="phone-row">
              <div class="input-group">
                <label>验证码</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg></span>
                  <input type="text" v-model="smsForm.code" placeholder="6位验证码" maxlength="6" @input="updateCaptchaState">
                </div>
                <div class="field-error" :class="{ show: errors.smsCode }">{{ errors.smsCode }}</div>
              </div>
              <button class="sms-btn" :disabled="smsCountdown > 0" @click="sendSmsCode">{{ smsCountdown > 0 ? `${smsCountdown}s 后重发` : '获取验证码' }}</button>
            </div>
          </div>

          <!-- ═══ 注册表单 ═══ -->
          <div v-if="currentMode === 'register'" class="form-section active">
            <div class="form-row">
              <div class="input-group">
                <label>用户名</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="8" r="4"/><path d="M6 21v-2a4 4 0 014-4h4a4 4 0 014 4v2"/></svg></span>
                  <input type="text" v-model="regForm.username" placeholder="请输入用户名" @input="updateCaptchaState">
                </div>
                <div class="field-error" :class="{ show: errors.regUsername }">{{ errors.regUsername }}</div>
              </div>
              <div class="input-group">
                <label>手机号</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="5" y="2" width="14" height="20" rx="2"/><line x1="12" y1="18" x2="12" y2="18.01"/></svg></span>
                  <input type="tel" v-model="regForm.phone" placeholder="请输入手机号" maxlength="11" @input="updateCaptchaState">
                </div>
                <div class="field-error" :class="{ show: errors.regPhone }">{{ errors.regPhone }}</div>
              </div>
            </div>
            <div class="form-row">
              <div class="input-group">
                <label>设置密码</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg></span>
                  <input :type="showRegPw ? 'text' : 'password'" v-model="regForm.password" placeholder="6-20位，含字母和数字" @input="onRegPwInput" autocomplete="new-password">
                  <button class="pw-toggle" @click="showRegPw = !showRegPw">{{ showRegPw ? '🙈' : '👁' }}</button>
                </div>
                <div class="pw-strength" v-if="regForm.password">
                  <span v-for="i in 4" :key="i" class="pw-strength-bar" :class="i <= pwStrength ? 'l' + Math.min(pwStrength, 4) : ''"></span>
                </div>
                <div class="field-error" :class="{ show: errors.regPassword }">{{ errors.regPassword }}</div>
              </div>
              <div class="input-group">
                <label>确认密码</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg></span>
                  <input :type="showRegPw2 ? 'text' : 'password'" v-model="regForm.confirmPassword" placeholder="请再次输入密码" @input="updateCaptchaState" autocomplete="new-password">
                  <button class="pw-toggle" @click="showRegPw2 = !showRegPw2">{{ showRegPw2 ? '🙈' : '👁' }}</button>
                </div>
                <div class="field-error" :class="{ show: errors.regPassword2 }">{{ errors.regPassword2 }}</div>
              </div>
            </div>
            <label class="agreement-row">
              <input type="checkbox" v-model="regForm.agreed" @change="updateCaptchaState">
              <span class="cb-box"></span> 我已阅读并同意 <a href="#" @click.prevent>《服务协议》</a> 和 <a href="#" @click.prevent>《隐私政策》</a>
            </label>
          </div>

          <!-- ═══ 企业登录 / 注册 ═══ -->
          <div v-if="currentMode === 'enterprise'" class="form-section active">
            <!-- 登录模式 -->
            <template v-if="entMode === 'login'">
              <div class="input-group">
                <label>企业编号 / 域名</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg></span>
                  <input type="text" v-model="entForm.domain" placeholder="请输入企业编号或域名" @input="updateCaptchaState">
                </div>
                <div class="field-error" :class="{ show: errors.entDomain }">{{ errors.entDomain }}</div>
              </div>
              <div class="input-group">
                <label>管理员账号</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="8" r="4"/><path d="M6 21v-2a4 4 0 014-4h4a4 4 0 014 4v2"/></svg></span>
                  <input type="text" v-model="entForm.username" placeholder="请输入管理员账号" @input="updateCaptchaState">
                </div>
                <div class="field-error" :class="{ show: errors.entUsername }">{{ errors.entUsername }}</div>
              </div>
              <div class="input-group">
                <label>密码</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg></span>
                  <input :type="showEntPw ? 'text' : 'password'" v-model="entForm.password" placeholder="请输入密码" @input="updateCaptchaState" @keyup.enter="handleEnterpriseLogin" autocomplete="current-password">
                  <button class="pw-toggle" @click="showEntPw = !showEntPw">{{ showEntPw ? '🙈' : '👁' }}</button>
                </div>
                <div class="field-error" :class="{ show: errors.entPassword }">{{ errors.entPassword }}</div>
              </div>
            </template>

            <!-- 注册模式 -->
            <template v-if="entMode === 'register'">
              <div class="input-group">
                <label>企业编号</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg></span>
                  <input type="text" v-model="entRegForm.domain" placeholder="请输入企业编号，如 a001" @input="updateCaptchaState">
                </div>
                <div class="field-error" :class="{ show: errors.entRegDomain }">{{ errors.entRegDomain }}</div>
              </div>
              <div class="input-group">
                <label>管理员账号</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="8" r="4"/><path d="M6 21v-2a4 4 0 014-4h4a4 4 0 014 4v2"/></svg></span>
                  <input type="text" v-model="entRegForm.username" placeholder="请输入管理员账号" @input="updateCaptchaState">
                </div>
                <div class="field-error" :class="{ show: errors.entRegUsername }">{{ errors.entRegUsername }}</div>
              </div>
              <div class="input-group">
                <label>设置密码</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg></span>
                  <input :type="showEntRegPw ? 'text' : 'password'" v-model="entRegForm.password" placeholder="6-20位，含字母和数字" @input="updateCaptchaState" autocomplete="new-password">
                  <button class="pw-toggle" @click="showEntRegPw = !showEntRegPw">{{ showEntRegPw ? '🙈' : '👁' }}</button>
                </div>
                <div class="field-error" :class="{ show: errors.entRegPassword }">{{ errors.entRegPassword }}</div>
              </div>
              <div class="input-group">
                <label>确认密码</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg></span>
                  <input :type="showEntRegPw2 ? 'text' : 'password'" v-model="entRegForm.confirmPassword" placeholder="请再次输入密码" @input="updateCaptchaState" autocomplete="new-password">
                  <button class="pw-toggle" @click="showEntRegPw2 = !showEntRegPw2">{{ showEntRegPw2 ? '🙈' : '👁' }}</button>
                </div>
                <div class="field-error" :class="{ show: errors.entRegPassword2 }">{{ errors.entRegPassword2 }}</div>
              </div>

              <!-- 授权验证分隔 -->
              <div style="border-top:1px solid rgba(168,85,247,0.12);margin:10px 0 12px;padding-top:8px;font-size:11px;color:#58638a;text-align:center">
                授权验证 — 需输入系统管理员账号确认操作权限
              </div>

              <div class="input-group">
                <label>授权管理员账号</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="8" r="4"/><path d="M6 21v-2a4 4 0 014-4h4a4 4 0 014 4v2"/></svg></span>
                  <input type="text" v-model="entRegForm.authAdmin" placeholder="请输入系统管理员账号" @input="updateCaptchaState">
                </div>
                <div class="field-error" :class="{ show: errors.entRegAuthAdmin }">{{ errors.entRegAuthAdmin }}</div>
              </div>
              <div class="input-group">
                <label>授权管理员密码</label>
                <div class="input-wrapper">
                  <span class="input-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg></span>
                  <input :type="showEntRegAuthPw ? 'text' : 'password'" v-model="entRegForm.authPassword" placeholder="请输入系统管理员密码" @input="updateCaptchaState" autocomplete="off">
                  <button class="pw-toggle" @click="showEntRegAuthPw = !showEntRegAuthPw">{{ showEntRegAuthPw ? '🙈' : '👁' }}</button>
                </div>
                <div class="field-error" :class="{ show: errors.entRegAuthPassword }">{{ errors.entRegAuthPassword }}</div>
              </div>
            </template>
          </div>

          <!-- ═══ 滑块验证码 ═══ -->
          <div class="captcha-section">
            <div class="captcha-track" :class="{ disabled: !captchaEnabled, success: captchaVerified }" ref="captchaTrackRef">
              <div class="captcha-progress" ref="captchaProgressRef"></div>
              <span class="track-text" ref="captchaTextRef">请按住滑块拖到最右侧完成验证</span>
              <div class="captcha-thumb" :class="{ success: captchaVerified }" ref="captchaThumbRef" @mousedown="onCaptchaStart" @touchstart.prevent="onCaptchaStart">→</div>
            </div>
            <div class="captcha-hint" :class="{ show: showCaptchaHint }">请先完整填写必填字段后再进行验证</div>
          </div>

          <!-- ═══ 按钮区 ═══ -->
          <div style="margin-top:16px">
            <!-- 登录按钮 -->
            <div v-if="currentMode === 'login'" class="btn-row">
              <button class="btn btn-primary" :class="{ loading: loading }" :disabled="loading" @click="handleLogin">
                <span class="btn-text">登 录</span><span class="spinner"></span>
              </button>
              <button class="btn btn-secondary" @click="switchMode('register')">
                <span class="btn-text">创建新账号</span>
              </button>
            </div>

            <!-- 注册按钮 -->
            <div v-if="currentMode === 'register'" class="btn-row">
              <button class="btn btn-primary" :class="{ loading: loading }" :disabled="loading" @click="handleRegister">
                <span class="btn-text">注 册</span><span class="spinner"></span>
              </button>
              <button class="btn btn-secondary" @click="switchMode('login')">
                <span class="btn-text">已有账号？登录</span>
              </button>
            </div>

            <!-- 企业登录/注册按钮 -->
            <div v-if="currentMode === 'enterprise' && entMode === 'login'" class="btn-row">
              <button class="btn btn-primary" :class="{ loading: loading }" :disabled="loading" @click="handleEnterpriseLogin">
                <span class="btn-text">企业登录</span><span class="spinner"></span>
              </button>
              <button class="btn btn-secondary" @click="switchEntMode('register')">
                <span class="btn-text">注册企业</span>
              </button>
            </div>
            <div v-if="currentMode === 'enterprise' && entMode === 'register'" class="btn-row">
              <button class="btn btn-primary" :class="{ loading: loading }" :disabled="loading" @click="handleEnterpriseRegister">
                <span class="btn-text">注 册</span><span class="spinner"></span>
              </button>
              <button class="btn btn-secondary" @click="switchEntMode('login')">
                <span class="btn-text">返回登录</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Toast 容器 -->
    <div class="toast-container">
      <div v-for="(t, i) in toasts" :key="i" class="toast" :class="'toast-' + t.type">
        <span class="toast-icon">{{ toastIcons[t.type] }}</span>
        <span>{{ t.message }}</span>
        <button class="toast-close" @click="toasts.splice(i, 1)">×</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { authApi } from '../api/auth'

const router = useRouter()
const auth = useAuthStore()

/* ═══════════════════════════════════════════════════════════════
   Canvas 粒子网络
   ═══════════════════════════════════════════════════════════════ */
const canvasRef = ref<HTMLCanvasElement | null>(null)
let animFrame = 0
let W = 0, H = 0
const mouse = { x: -999, y: -999 }
const PARTICLE_COUNT = 100
const CONNECT_DIST = 150
const MOUSE_REPEL_DIST = 120
const MOUSE_ATTRACT_DIST = 180

function initCanvas() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')!
  W = canvas.width = window.innerWidth
  H = canvas.height = window.innerHeight

  const particles = Array.from({ length: PARTICLE_COUNT }, () => ({
    x: Math.random() * W, y: Math.random() * H,
    vx: (Math.random() - 0.5) * 0.5, vy: (Math.random() - 0.5) * 0.5,
    size: 1.2 + Math.random() * 1.8,
  }))

  function animate() {
    ctx.fillStyle = 'rgba(5, 8, 22, 0.18)'
    ctx.fillRect(0, 0, W, H)

    particles.forEach(p => {
      const dx = p.x - mouse.x, dy = p.y - mouse.y
      const dist = Math.sqrt(dx * dx + dy * dy)
      if (dist < MOUSE_REPEL_DIST && dist > 0) {
        const force = (MOUSE_REPEL_DIST - dist) / MOUSE_REPEL_DIST
        p.vx += (dx / dist) * force * 0.8
        p.vy += (dy / dist) * force * 0.8
      }
      p.vx *= 0.98; p.vy *= 0.98
      p.x += p.vx; p.y += p.vy
      if (p.x < 0) { p.x = 0; p.vx *= -1 }
      if (p.x > W) { p.x = W; p.vx *= -1 }
      if (p.y < 0) { p.y = 0; p.vy *= -1 }
      if (p.y > H) { p.y = H; p.vy *= -1 }
      ctx.beginPath()
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(168, 133, 247, 0.6)'
      ctx.fill()
    })

    // 粒子间连线
    ctx.strokeStyle = 'rgba(168, 133, 247, 0.12)'
    ctx.lineWidth = 0.6
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x, dy = particles[i].y - particles[j].y
        const d = Math.sqrt(dx * dx + dy * dy)
        if (d < CONNECT_DIST) {
          ctx.beginPath()
          ctx.moveTo(particles[i].x, particles[i].y)
          ctx.lineTo(particles[j].x, particles[j].y)
          ctx.globalAlpha = 1 - d / CONNECT_DIST
          ctx.stroke()
          ctx.globalAlpha = 1
        }
      }
    }

    // 鼠标连线
    if (mouse.x > 0 && mouse.y > 0) {
      ctx.strokeStyle = 'rgba(192, 132, 252, 0.28)'
      ctx.lineWidth = 1
      particles.forEach(p => {
        const dx = p.x - mouse.x, dy = p.y - mouse.y
        const d = Math.sqrt(dx * dx + dy * dy)
        if (d < MOUSE_ATTRACT_DIST) {
          ctx.beginPath()
          ctx.moveTo(p.x, p.y)
          ctx.lineTo(mouse.x, mouse.y)
          ctx.globalAlpha = 1 - d / MOUSE_ATTRACT_DIST
          ctx.stroke()
          ctx.globalAlpha = 1
        }
      })
    }
    animFrame = requestAnimationFrame(animate)
  }
  animate()
}

function onMouseMove(e: MouseEvent) { mouse.x = e.clientX; mouse.y = e.clientY }
function onMouseLeave() { mouse.x = -999; mouse.y = -999 }
function onTouchMove(e: TouchEvent) { const t = e.touches[0]; mouse.x = t.clientX; mouse.y = t.clientY }
function onTouchEnd() { mouse.x = -999; mouse.y = -999 }

function onResize() {
  if (canvasRef.value) {
    W = canvasRef.value.width = window.innerWidth
    H = canvasRef.value.height = window.innerHeight
  }
}

onMounted(() => {
  nextTick(initCanvas)
  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseleave', onMouseLeave)
  window.addEventListener('touchmove', onTouchMove, { passive: true })
  window.addEventListener('touchend', onTouchEnd)
  window.addEventListener('resize', onResize)
})

onUnmounted(() => {
  cancelAnimationFrame(animFrame)
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseleave', onMouseLeave)
  window.removeEventListener('touchmove', onTouchMove)
  window.removeEventListener('touchend', onTouchEnd)
  window.removeEventListener('resize', onResize)
})

/* ═══════════════════════════════════════════════════════════════
   Auth 状态管理
   ═══════════════════════════════════════════════════════════════ */
const currentMode = ref<'login' | 'register' | 'enterprise'>('login')
const loginTab = ref<'password' | 'sms'>('password')
const captchaVerified = ref(false)
const showCaptchaHint = ref(false)
const loading = ref(false)

const showLoginPw = ref(false)
const showRegPw = ref(false)
const showRegPw2 = ref(false)
const showEntPw = ref(false)
const rememberMe = ref(false)

const loginForm = reactive({ username: '', password: '' })
const smsForm = reactive({ phone: '', code: '' })
const regForm = reactive({ username: '', phone: '', password: '', confirmPassword: '', agreed: false })
const entForm = reactive({ domain: '', username: '', password: '' })
const entMode = ref<'login' | 'register'>('login')
const entRegForm = reactive({ domain: '', username: '', password: '', confirmPassword: '', authAdmin: '', authPassword: '' })
const showEntRegPw = ref(false)
const showEntRegPw2 = ref(false)
const showEntRegAuthPw = ref(false)

const errors = reactive<Record<string, string>>({})
const smsCountdown = ref(0)
let smsTimer: ReturnType<typeof setInterval> | null = null

// Toast
const toasts = ref<{ type: 'success' | 'error' | 'info'; message: string }[]>([])
const toastIcons = { success: '✓', error: '✕', info: 'ℹ' }
let toastId = 0

function showToast(type: 'success' | 'error' | 'info', message: string) {
  const t = { type, message }
  toasts.value.push(t)
  setTimeout(() => {
    const idx = toasts.value.indexOf(t)
    if (idx >= 0) toasts.value.splice(idx, 1)
  }, 3500)
}

function clearErrors() {
  Object.keys(errors).forEach(k => delete errors[k])
}

/* ── 模式切换 ── */
function switchMode(mode: typeof currentMode.value) {
  currentMode.value = mode
  captchaVerified.value = false
  resetCaptcha()
  clearErrors()
  if (mode === 'enterprise') entMode.value = 'login'
}

function switchLoginTab(tab: typeof loginTab.value) {
  loginTab.value = tab
  clearErrors()
  updateCaptchaState()
}

function switchEntMode(mode: 'login' | 'register') {
  entMode.value = mode
  captchaVerified.value = false
  resetCaptcha()
  clearErrors()
}

/* ═══════════════════════════════════════════════════════════════
   滑块验证码
   ═══════════════════════════════════════════════════════════════ */
const captchaTrackRef = ref<HTMLElement | null>(null)
const captchaThumbRef = ref<HTMLElement | null>(null)
const captchaProgressRef = ref<HTMLElement | null>(null)
const captchaTextRef = ref<HTMLElement | null>(null)

let captchaDragging = false
let captchaStartX = 0, captchaStartLeft = 0, captchaMaxLeft = 0

function getMaxLeft() {
  const track = captchaTrackRef.value, thumb = captchaThumbRef.value
  if (!track || !thumb) return 0
  return track.clientWidth - thumb.clientWidth - 4
}

const captchaEnabled = computed(() => {
  if (currentMode.value === 'login') {
    if (loginTab.value === 'password') {
      return loginForm.username.trim() !== '' && loginForm.password.trim() !== ''
    }
    return smsForm.phone.trim() !== '' && smsForm.code.trim().length >= 4
  }
  if (currentMode.value === 'register') {
    return regForm.username.trim() !== '' && regForm.phone.trim() !== '' &&
           regForm.password.trim() !== '' && regForm.confirmPassword.trim() !== '' && regForm.agreed
  }
  if (currentMode.value === 'enterprise') {
    if (entMode.value === 'login') {
      return entForm.domain.trim() !== '' && entForm.username.trim() !== '' && entForm.password.trim() !== ''
    }
    return entRegForm.domain.trim() !== '' && entRegForm.username.trim() !== '' &&
           entRegForm.password.trim() !== '' && entRegForm.confirmPassword.trim() !== '' &&
           entRegForm.authAdmin.trim() !== '' && entRegForm.authPassword.trim() !== ''
  }
  return false
})

function updateCaptchaState() {
  showCaptchaHint.value = false
  if (!captchaEnabled.value && captchaVerified.value) {
    captchaVerified.value = false
    resetCaptcha()
  }
}

function resetCaptcha() {
  const thumb = captchaThumbRef.value, progress = captchaProgressRef.value, text = captchaTextRef.value
  if (thumb) { thumb.style.left = '2px'; thumb.textContent = '→' }
  if (progress) progress.style.width = '0'
  if (text) { text.textContent = '请按住滑块拖到最右侧完成验证'; text.style.opacity = '1' }
}

function onCaptchaStart(e: MouseEvent | TouchEvent) {
  if (!captchaEnabled.value) {
    showCaptchaHint.value = true
    setTimeout(() => showCaptchaHint.value = false, 2000)
    return
  }
  if (captchaVerified.value) return
  captchaDragging = true
  captchaMaxLeft = getMaxLeft()
  const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX
  captchaStartX = clientX
  captchaStartLeft = parseInt(captchaThumbRef.value?.style.left || '2')
}

function onCaptchaMove(e: MouseEvent | TouchEvent) {
  if (!captchaDragging) return
  const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX
  let newLeft = captchaStartLeft + (clientX - captchaStartX)
  newLeft = Math.max(2, Math.min(captchaMaxLeft, newLeft))
  if (captchaThumbRef.value) captchaThumbRef.value.style.left = newLeft + 'px'
  if (captchaProgressRef.value) captchaProgressRef.value.style.width = (newLeft + 19) + 'px'
}

function onCaptchaEnd() {
  if (!captchaDragging) return
  captchaDragging = false
  const left = parseInt(captchaThumbRef.value?.style.left || '2')
  if (left >= captchaMaxLeft * 0.9) {
    if (captchaThumbRef.value) { captchaThumbRef.value.style.left = captchaMaxLeft + 'px'; captchaThumbRef.value.textContent = '✓' }
    if (captchaProgressRef.value) captchaProgressRef.value.style.width = '100%'
    if (captchaTextRef.value) captchaTextRef.value.textContent = '验证通过'
    captchaVerified.value = true
  } else {
    const thumb = captchaThumbRef.value, progress = captchaProgressRef.value
    if (thumb) { thumb.style.transition = 'left 0.35s cubic-bezier(0.4,0,0.2,1)'; thumb.style.left = '2px' }
    if (progress) { progress.style.transition = 'width 0.35s cubic-bezier(0.4,0,0.2,1)'; progress.style.width = '0' }
    setTimeout(() => {
      if (thumb) thumb.style.transition = ''
      if (progress) progress.style.transition = ''
    }, 350)
  }
}

onMounted(() => {
  document.addEventListener('mousemove', onCaptchaMove)
  document.addEventListener('mouseup', onCaptchaEnd)
  document.addEventListener('touchmove', onCaptchaMove, { passive: false })
  document.addEventListener('touchend', onCaptchaEnd)
})

onUnmounted(() => {
  document.removeEventListener('mousemove', onCaptchaMove)
  document.removeEventListener('mouseup', onCaptchaEnd)
  document.removeEventListener('touchmove', onCaptchaMove)
  document.removeEventListener('touchend', onCaptchaEnd)
})

/* ═══════════════════════════════════════════════════════════════
   校验
   ═══════════════════════════════════════════════════════════════ */
function validatePhone(phone: string) { return /^1[3-9]\d{9}$/.test(phone) }
function validatePassword(pw: string) { return pw.length >= 6 && pw.length <= 20 && /[a-zA-Z]/.test(pw) && /\d/.test(pw) }

const pwStrength = computed(() => {
  const pw = regForm.password
  if (pw.length < 6) return 0
  let s = 0
  if (pw.length >= 8) s++
  if (/[a-z]/.test(pw) && /[A-Z]/.test(pw)) s++
  else if (/[a-zA-Z]/.test(pw)) s += 0.5
  if (/\d/.test(pw)) s++
  if (/[^a-zA-Z0-9]/.test(pw)) s++
  return Math.min(4, Math.floor(s))
})

function onRegPwInput() { updateCaptchaState() }

/* ═══════════════════════════════════════════════════════════════
   API 操作
   ═══════════════════════════════════════════════════════════════ */
async function handleLogin() {
  clearErrors()
  let ok = true

  if (loginTab.value === 'password') {
    if (!loginForm.username) { errors.loginUsername = '请输入账号或手机号'; ok = false }
    if (!loginForm.password) { errors.loginPassword = '请输入密码'; ok = false }
    if (!captchaVerified.value) { showCaptchaHint.value = true; ok = false }
    if (!ok) return

    loading.value = true
    try {
      await auth.login(loginForm.username, loginForm.password)
      showToast('success', `欢迎回来，${auth.user?.username}！`)
      setTimeout(() => {
        if (auth.isMerchant()) router.push('/admin')
        else router.push('/chat')
      }, 600)
    } catch {
      // 错误已由 axios 拦截器 ElMessage 处理
    } finally { loading.value = false }
  } else {
    if (!validatePhone(smsForm.phone)) { errors.smsPhone = '请输入有效的手机号'; ok = false }
    if (smsForm.code.length < 4) { errors.smsCode = '请输入验证码'; ok = false }
    if (!captchaVerified.value) { showCaptchaHint.value = true; ok = false }
    if (!ok) return

    loading.value = true
    try {
      await auth.login(smsForm.phone, smsForm.code)
      showToast('success', `欢迎回来，${auth.user?.username}！`)
      setTimeout(() => {
        if (auth.isMerchant()) router.push('/admin')
        else router.push('/chat')
      }, 600)
    } catch {
      // 错误已由 axios 拦截器 ElMessage 处理
    } finally { loading.value = false }
  }
}

async function handleRegister() {
  clearErrors()
  let ok = true

  if (!regForm.username) { errors.regUsername = '请输入用户名'; ok = false }
  if (!validatePhone(regForm.phone)) { errors.regPhone = '请输入有效的手机号'; ok = false }
  if (!validatePassword(regForm.password)) { errors.regPassword = '密码需6-20位，含字母和数字'; ok = false }
  if (regForm.password !== regForm.confirmPassword) { errors.regPassword2 = '两次输入的密码不一致'; ok = false }
  if (!regForm.agreed) { showToast('info', '请先阅读并同意服务协议和隐私政策'); ok = false }
  if (!captchaVerified.value) { showCaptchaHint.value = true; ok = false }
  if (!ok) return

  loading.value = true
  try {
    await authApi.register(regForm.username, regForm.password, regForm.phone)
    showToast('success', '注册成功！请登录')
    switchMode('login')
    loginForm.username = regForm.username
    loginForm.password = ''
  } catch {
    // 错误已由 axios 拦截器 ElMessage 处理
  } finally { loading.value = false }
}

async function handleEnterpriseLogin() {
  clearErrors()
  let ok = true

  if (!entForm.domain) { errors.entDomain = '请输入企业编号或域名'; ok = false }
  if (!entForm.username) { errors.entUsername = '请输入管理员账号'; ok = false }
  if (!entForm.password) { errors.entPassword = '请输入密码'; ok = false }
  if (!captchaVerified.value) { showCaptchaHint.value = true; ok = false }
  if (!ok) return

  loading.value = true
  try {
    const loginName = `${entForm.domain}_${entForm.username}`
    await auth.login(loginName, entForm.password)
    showToast('success', `企业登录成功，${auth.user?.username}！`)
    setTimeout(() => {
      if (auth.isMerchant()) router.push('/admin')
      else router.push('/chat')
    }, 600)
  } catch {
    // 错误已由 axios 拦截器 ElMessage 处理
  } finally { loading.value = false }
}

async function handleEnterpriseRegister() {
  clearErrors()
  let ok = true

  if (!entRegForm.domain) { errors.entRegDomain = '请输入企业编号'; ok = false }
  if (!entRegForm.username) { errors.entRegUsername = '请输入管理员账号'; ok = false }
  if (!validatePassword(entRegForm.password)) { errors.entRegPassword = '密码需6-20位，含字母和数字'; ok = false }
  if (entRegForm.password !== entRegForm.confirmPassword) { errors.entRegPassword2 = '两次输入的密码不一致'; ok = false }
  if (!entRegForm.authAdmin) { errors.entRegAuthAdmin = '请输入系统管理员账号'; ok = false }
  if (!entRegForm.authPassword) { errors.entRegAuthPassword = '请输入系统管理员密码'; ok = false }
  if (!captchaVerified.value) { showCaptchaHint.value = true; ok = false }
  if (!ok) return

  loading.value = true
  try {
    // 第一步：验证管理员身份（仅调API，不污染store）
    let adminData: any
    try {
      adminData = await authApi.login(entRegForm.authAdmin, entRegForm.authPassword)
    } catch {
      showToast('error', '管理员账号或密码错误，无权创建企业')
      loading.value = false
      return
    }

    if (adminData.role !== 'admin') {
      showToast('error', '该账号不是系统管理员，无权创建企业')
      loading.value = false
      return
    }

    // 第二步：创建企业账号
    const loginName = `${entRegForm.domain}_${entRegForm.username}`
    await authApi.register(loginName, entRegForm.password, undefined, 'merchant', entRegForm.domain)
    showToast('success', '企业注册成功！请登录')
    entForm.domain = entRegForm.domain
    entForm.username = entRegForm.username
    entForm.password = ''
    switchEntMode('login')
  } catch {
    // 注册失败，错误由 axios 拦截器处理
  } finally { loading.value = false }
}

/* ═══════════════════════════════════════════════════════════════
   SMS 倒计时
   ═══════════════════════════════════════════════════════════════ */
function sendSmsCode() {
  if (!validatePhone(smsForm.phone)) { errors.smsPhone = '请输入有效的手机号'; return }
  if (smsCountdown.value > 0) return

  smsCountdown.value = 60
  showToast('info', `验证码已发送至 ${smsForm.phone}（演示模式：输入任意6位数字）`)

  smsTimer = setInterval(() => {
    smsCountdown.value--
    if (smsCountdown.value <= 0) {
      if (smsTimer) clearInterval(smsTimer)
    }
  }, 1000)
}

onUnmounted(() => {
  if (smsTimer) clearInterval(smsTimer)
})
</script>

<style>
/* ═══════════════════════════════════════════════════════════════
   设计系统：暗色科技风登录页 — 全局非 scoped
   （仅在 LoginView 激活时生效，不污染其他页面）
   ═══════════════════════════════════════════════════════════════ */
.login-page {
  --bg-deep: #050816;
  --bg-mid: #0b142f;
  --bg-card: rgba(11, 20, 47, 0.65);
  --accent: #a855f7;
  --accent-light: #c084fc;
  --accent-glow: rgba(168, 85, 247, 0.35);
  --accent-subtle: rgba(168, 85, 247, 0.12);
  --text: #f0f4ff;
  --text-secondary: #8892b0;
  --text-dim: #58638a;
  --border: rgba(168, 85, 247, 0.15);
  --border-focus: rgba(168, 85, 247, 0.55);
  --red: #ef4444;
  --red-glow: rgba(239, 68, 68, 0.3);
  --green: #22c55e;
  --green-glow: rgba(34, 197, 94, 0.35);
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --font-display: 'Chakra Petch', system-ui, sans-serif;
  --font-body: 'Sora', 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif;

  position: fixed; inset: 0;
  display: flex; align-items: center; justify-content: center;
  background: var(--bg-deep);
  font-family: var(--font-body);
  font-size: 14px;
  color: var(--text);
  -webkit-font-smoothing: antialiased;
  overflow: hidden;
  z-index: 999;
}

.login-page canvas {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
}

.bg-orb {
  position: fixed; border-radius: 50%; filter: blur(120px);
  pointer-events: none; z-index: 0; opacity: 0.4;
}
.bg-orb-1 { width: 500px; height: 500px; background: radial-gradient(circle, #a855f7 0%, transparent 70%); top: -15%; right: -10%; animation: orbFloat1 12s ease-in-out infinite; }
.bg-orb-2 { width: 350px; height: 350px; background: radial-gradient(circle, #6366f1 0%, transparent 70%); bottom: -10%; left: -5%;  animation: orbFloat2 15s ease-in-out infinite; }
.bg-orb-3 { width: 250px; height: 250px; background: radial-gradient(circle, #c084fc 0%, transparent 70%); top: 40%; left: 40%; animation: orbFloat3 10s ease-in-out infinite; }
@keyframes orbFloat1 { 0%,100%{transform:translate(0,0)} 33%{transform:translate(-40px,30px)} 66%{transform:translate(20px,-20px)} }
@keyframes orbFloat2 { 0%,100%{transform:translate(0,0)} 33%{transform:translate(30px,-30px)} 66%{transform:translate(-20px,20px)} }
@keyframes orbFloat3 { 0%,100%{transform:translate(0,0)scale(1)} 50%{transform:translate(-20px,-20px)scale(1.15)} }

.grid-bg {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(168,85,247,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(168,85,247,0.04) 1px, transparent 1px);
  background-size: 60px 60px;
  mask-image: radial-gradient(ellipse at center, black 30%, transparent 70%);
}

.main-container {
  position: relative; z-index: 1;
  display: flex;
  width: 100vw; height: 100vh;
  max-width: 1280px; max-height: 800px;
  margin: auto; padding: 0 40px; gap: 0;
  animation: containerIn 0.8s cubic-bezier(0.2,0.9,0.3,1) forwards;
}
@keyframes containerIn { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }

/* ── 左侧品牌区 ── */
.brand-panel {
  flex: 0 0 44%; display: flex; flex-direction: column;
  justify-content: center; padding-right: 30px; z-index: 1;
}
.brand-badge {
  display: inline-flex; align-items: center; gap: 10px;
  font-family: var(--font-display); font-size: 13px; font-weight: 500;
  color: #c084fc; letter-spacing: 3px; text-transform: uppercase;
  margin-bottom: 28px; padding: 6px 14px;
  border: 1px solid rgba(168,85,247,0.25); border-radius: 20px;
  background: rgba(168,85,247,0.08);
}
.brand-badge .dot { width: 7px; height: 7px; background: #a855f7; border-radius: 50%; box-shadow: 0 0 8px #a855f7; animation: dotPulse 2s infinite; }
@keyframes dotPulse { 0%,100%{box-shadow:0 0 8px #a855f7} 50%{box-shadow:0 0 18px #c084fc} }

.brand-title {
  font-family: var(--font-display);
  font-size: clamp(36px, 4.5vw, 52px); font-weight: 700;
  line-height: 1.15; margin-bottom: 12px; letter-spacing: -0.5px;
}
.brand-title .highlight { background: linear-gradient(135deg, #c084fc, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.brand-desc { font-size: 15px; color: #8892b0; line-height: 1.7; margin-bottom: 40px; max-width: 380px; }
.feature-cards { display: flex; flex-direction: column; gap: 14px; }
.feature-card {
  display: flex; align-items: center; gap: 14px;
  padding: 14px 18px; border-radius: var(--radius-md);
  background: rgba(168,85,247,0.04); border: 1px solid rgba(168,85,247,0.08);
  transition: all 0.35s ease; cursor: default;
}
.feature-card:hover {
  background: rgba(168,85,247,0.09); border-color: rgba(168,85,247,0.22);
  transform: translateX(4px);
}
.feature-card .fc-icon { width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; background: rgba(168,85,247,0.12); flex-shrink: 0; }
.feature-card .fc-text h4 { font-size: 14px; font-weight: 600; margin-bottom: 2px; }
.feature-card .fc-text p  { font-size: 12px; color: #8892b0; }
.stats-row { display: flex; gap: 32px; margin-top: 36px; }
.stat-item { text-align: left; }
.stat-num { font-family: var(--font-display); font-size: 28px; font-weight: 700; color: #c084fc; }
.stat-num .stat-unit { font-size: 16px; }
.stat-label { font-size: 12px; color: #8892b0; margin-top: 2px; }

/* ── 右侧认证卡片 ── */
.auth-panel { flex: 0 0 56%; display: flex; align-items: center; justify-content: center; z-index: 1; }
.auth-card {
  width: 100%; max-width: 520px;
  background: var(--bg-card);
  backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
  border: 1px solid var(--border); border-radius: var(--radius-lg);
  padding: 32px 36px 28px; position: relative;
  box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
  animation: cardIn 0.7s 0.15s cubic-bezier(0.2,0.9,0.3,1) both;
}
@keyframes cardIn { from{opacity:0;transform:translateX(24px) scale(0.98)} to{opacity:1;transform:translateX(0) scale(1)} }
.auth-card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
.auth-card-title { font-family: var(--font-display); font-size: 22px; font-weight: 600; letter-spacing: 0.5px; color: #f0f4ff; }
.enterprise-link {
  font-family: var(--font-body); font-size: 12.5px; color: #8892b0;
  cursor: pointer; display: flex; align-items: center; gap: 4px;
  transition: color 0.25s; border: none; background: none; padding: 4px 0; line-height: 1;
}
.enterprise-link:hover { color: #c084fc; }
.enterprise-link svg { width: 13px; height: 13px; transition: transform 0.25s; }
.enterprise-link:hover svg { transform: translateX(3px); }

.login-tabs { display: flex; position: relative; margin-bottom: 20px; border-radius: var(--radius-sm); background: rgba(255,255,255,0.03); padding: 3px; }
.login-tab { flex:1; text-align:center; padding:9px 0; font-size:13px; font-weight:500; color:#8892b0; cursor:pointer; border-radius:6px; transition:color .3s; position:relative; z-index:1; border:none; background:none; font-family:var(--font-body); }
.login-tab.active { color:#fff; }
.tab-slider { position:absolute; top:3px; left:3px; width:calc(50% - 3px); height:calc(100% - 6px); background:rgba(168,85,247,0.2); border-radius:6px; transition:transform .35s cubic-bezier(0.4,0,0.2,1); z-index:0; }
.tab-slider.right { transform:translateX(100%); }

.form-section { display: none; }
.form-section.active { display: block; animation: formIn .35s ease; }
@keyframes formIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
.form-row { display: flex; gap: 12px; }

.input-group { position: relative; margin-bottom: 14px; }
.input-group label { display: block; font-size: 12px; color: #8892b0; margin-bottom: 6px; transition: color 0.25s; font-weight: 500; }
.input-group:focus-within label { color: #c084fc; }
.input-wrapper { position: relative; display: flex; align-items: center; }
.input-wrapper input {
  width: 100%; padding: 11px 14px; padding-left: 38px;
  font-size: 14px; font-family: var(--font-body); color: #f0f4ff;
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
  border-radius: var(--radius-sm); outline: none; transition: all 0.3s;
}
.input-wrapper input:focus {
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px var(--accent-subtle), 0 0 14px var(--accent-glow);
  background: rgba(255,255,255,0.05);
}
.input-icon { position: absolute; left: 12px; top: 50%; transform: translateY(-50%); color: #58638a; font-size: 16px; transition: color 0.25s; pointer-events: none; display: flex; align-items: center; }
.input-group:focus-within .input-icon { color: #c084fc; }
.pw-toggle { position: absolute; right: 10px; top: 50%; transform: translateY(-50%); background: none; border: none; color: #58638a; cursor: pointer; padding: 4px; font-size: 15px; display: flex; align-items: center; transition: color 0.25s; }
.pw-toggle:hover { color: #c084fc; }

.phone-row { display: flex; gap: 10px; }
.phone-row .input-group { flex: 1; }
.sms-btn {
  flex-shrink: 0; align-self: flex-start; margin-top: 20px;
  padding: 11px 16px; font-size: 12.5px; font-weight: 500;
  white-space: nowrap; border-radius: var(--radius-sm);
  border: 1px solid #a855f7; background: transparent; color: #c084fc;
  cursor: pointer; font-family: var(--font-body); transition: all 0.3s;
}
.sms-btn:hover:not(:disabled) { background: #a855f7; color: #fff; box-shadow: 0 0 20px rgba(168,85,247,0.35); }
.sms-btn:disabled { border-color: #58638a; color: #58638a; cursor: not-allowed; }

.pw-strength { display: flex; gap: 5px; margin-top: 6px; }
.pw-strength-bar { flex:1; height:3px; border-radius:2px; background:rgba(255,255,255,0.06); transition:background .3s; }
.pw-strength-bar.l1{background:#ef4444}
.pw-strength-bar.l2{background:#f59e0b}
.pw-strength-bar.l3{background:#22c55e}
.pw-strength-bar.l4{background:#c084fc}

.field-error { font-size: 11px; color: #ef4444; margin-top: 4px; display: none; align-items: center; gap: 4px; }
.field-error.show { display: flex; }

.form-options { display: flex; align-items: center; justify-content: space-between; margin: 6px 0 16px; }
.remember-me { display: flex; align-items: center; gap: 7px; font-size: 12.5px; color: #8892b0; cursor: pointer; }
.remember-me input[type="checkbox"] { display: none; }
.remember-me .cb-box {
  width: 16px; height: 16px; border-radius: 4px;
  border: 1.5px solid rgba(255,255,255,0.15);
  display: flex; align-items: center; justify-content: center;
  transition: all 0.25s; flex-shrink: 0;
}
.remember-me input:checked + .cb-box { background: #a855f7; border-color: #a855f7; }
.remember-me input:checked + .cb-box::after { content: '✓'; color: #fff; font-size: 10px; font-weight: 700; }
.forgot-link { font-size: 12.5px; color: #8892b0; cursor: pointer; transition: color .25s; }
.forgot-link:hover { color: #c084fc; }

.agreement-row { display: flex; align-items: center; gap: 7px; font-size: 12px; color: #8892b0; margin-bottom: 16px; }
.agreement-row input[type="checkbox"] { display: none; }
.agreement-row .cb-box {
  width: 16px; height: 16px; border-radius: 4px;
  border: 1.5px solid rgba(255,255,255,0.15);
  display: flex; align-items: center; justify-content: center;
  transition: all 0.25s; flex-shrink: 0;
}
.agreement-row input:checked + .cb-box { background: #a855f7; border-color: #a855f7; }
.agreement-row input:checked + .cb-box::after { content: '✓'; color: #fff; font-size: 10px; font-weight: 700; }
.agreement-row a { color: #c084fc; text-decoration: none; }
.agreement-row a:hover { text-decoration: underline; }

.btn-row { display: flex; gap: 12px; margin-bottom: 16px; }
.btn {
  flex: 1; padding: 12px 0; font-size: 14px; font-weight: 600;
  font-family: var(--font-body); border-radius: var(--radius-sm);
  border: none; cursor: pointer; transition: all 0.35s;
  position: relative; overflow: hidden;
  display: flex; align-items: center; justify-content: center; gap: 8px;
}
.btn-primary { background: linear-gradient(135deg, #a855f7, #7c3aed); color: #fff; box-shadow: 0 4px 20px rgba(168,85,247,0.3); }
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(168,85,247,0.45); }
.btn-primary::after {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(105deg, transparent 40%, rgba(255,255,255,0.12) 45%, rgba(255,255,255,0.18) 50%, transparent 55%);
  transform: translateX(-100%); transition: transform 0.6s;
}
.btn-primary:hover::after { transform: translateX(100%); }
.btn-secondary { background: rgba(255,255,255,0.04); color: #f0f4ff; border: 1px solid rgba(255,255,255,0.1); }
.btn-secondary:hover { background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.2); transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.2); }
.btn:disabled { opacity: 0.45; cursor: not-allowed; pointer-events: none; }

.spinner { width: 18px; height: 18px; border: 2px solid rgba(255,255,255,0.3); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; display: none; }
.btn.loading .spinner { display: block; }
.btn.loading .btn-text { display: none; }
@keyframes spin { to{transform:rotate(360deg)} }

.back-link { display: inline-flex; align-items: center; gap: 4px; font-size: 12.5px; color: #8892b0; cursor: pointer; transition: color 0.25s; margin-bottom: 16px; border: none; background: none; font-family: var(--font-body); }
.back-link:hover { color: #c084fc; }

/* ── 滑块验证码 ── */
.captcha-section { margin-top: 6px; }
.captcha-track {
  position: relative; width: 100%; height: 44px;
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 22px; overflow: hidden; user-select: none; transition: all 0.4s;
}
.captcha-track.disabled { opacity: 0.4; filter: saturate(0.3); cursor: not-allowed; }
.captcha-track.success { border-color: #22c55e; background: rgba(34,197,94,0.06); }
.captcha-track .track-text { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 12.5px; color: #8892b0; pointer-events: none; transition: opacity 0.3s; }
.captcha-track.success .track-text { color: #22c55e; }
.captcha-progress { position: absolute; left: 0; top: 0; height: 100%; width: 0; border-radius: 22px; background: linear-gradient(90deg, rgba(34,197,94,0.15), rgba(34,197,94,0.3)); transition: width 0.05s linear; pointer-events: none; }
.captcha-thumb {
  position: absolute; left: 2px; top: 2px; width: 38px; height: 38px; border-radius: 50%;
  background: linear-gradient(135deg, #a855f7, #7c3aed);
  display: flex; align-items: center; justify-content: center;
  cursor: grab; box-shadow: 0 2px 12px rgba(168,85,247,0.5);
  transition: box-shadow 0.3s, background 0.4s; z-index: 2;
  color: #fff; font-size: 18px; font-weight: 700;
}
.captcha-thumb:active { cursor: grabbing; }
.captcha-thumb.success { background: #22c55e; box-shadow: 0 2px 12px rgba(34,197,94,0.35); }
.captcha-track.disabled .captcha-thumb { cursor: not-allowed; pointer-events: none; }
.captcha-hint { font-size: 11px; color: #ef4444; margin-top: 4px; text-align: center; display: none; }
.captcha-hint.show { display: block; }

/* ── Toast ── */
.toast-container { position: fixed; top: 24px; right: 24px; z-index: 99999; display: flex; flex-direction: column; gap: 8px; }
.toast {
  display: flex; align-items: center; gap: 10px;
  padding: 12px 20px; border-radius: var(--radius-md);
  font-size: 13px; font-weight: 500;
  backdrop-filter: blur(16px); box-shadow: 0 8px 28px rgba(0,0,0,0.4);
  animation: toastIn 0.4s cubic-bezier(0.2,0.9,0.3,1);
  min-width: 240px; max-width: 420px;
}
.toast-success { background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.3); color: #4ade80; }
.toast-error   { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); color: #f87171; }
.toast-info    { background: rgba(168,85,247,0.12); border: 1px solid rgba(168,85,247,0.25); color: #c084fc; }
.toast-icon { font-size: 18px; flex-shrink: 0; }
.toast-close { margin-left: auto; background: none; border: none; color: inherit; opacity: 0.6; cursor: pointer; font-size: 16px; padding: 0 2px; }
@keyframes toastIn { from{opacity:0;transform:translateX(80px)} to{opacity:1;transform:translateX(0)} }

/* ── 响应式 ── */
@media (max-width: 860px) {
  .main-container { flex-direction: column; max-height: none; padding: 20px; overflow-y: auto; }
  .brand-panel { flex: none; padding-right: 0; text-align: center; align-items: center; padding-top: 30px; }
  .brand-desc { max-width: 100%; }
  .feature-cards { display: none; }
  .stats-row { justify-content: center; }
  .auth-panel { flex: none; width: 100%; max-width: 440px; margin: 20px auto 40px; }
  .auth-card { padding: 24px 22px 22px; }
  .form-row { flex-direction: column; gap: 0; }
}
</style>