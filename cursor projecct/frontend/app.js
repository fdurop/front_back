const STORAGE_KEY = 'qa_messages';
const MOCK_KEY = 'qa_mock_mode';
let API_ENDPOINT = '/api/ask';
const isFileMode = typeof location !== 'undefined' && location.protocol === 'file:';

function getCandidateEndpoints() {
  const cached = localStorage.getItem('qa_api_endpoint') || '';
  if (!isFileMode) {
    return [
      cached,
      '/api/ask',
      'http://127.0.0.1:8000/api/ask',
      'http://localhost:8000/api/ask'
    ].filter(Boolean);
  }
  return [
    cached,
    'http://127.0.0.1:8000/api/ask',
    'http://localhost:8000/api/ask',
    'http://127.0.0.1:5000/api/ask',
    'http://localhost:5000/api/ask'
  ].filter(Boolean);
}

async function findWorkingEndpoint() {
  const candidates = getCandidateEndpoints();
  for (const endpoint of candidates) {
    const healthUrl = endpoint.replace('/api/ask', '/health');
    try {
      const res = await fetch(healthUrl, { method: 'GET' });
      if (res.ok) {
        localStorage.setItem('qa_api_endpoint', endpoint);
        API_ENDPOINT = endpoint;
        setBackendStatus(true, endpoint);
        return endpoint;
      }
    } catch (_) {}
  }
  setBackendStatus(false, null);
  return null;
}

async function updateBackendHealth() {
  const ok = await findWorkingEndpoint();
  if (!ok) setBackendStatus(false, null);
}

function setBackendStatus(connected, endpoint) {
  const backendStatus = document.getElementById('backendStatus');
  if (!backendStatus) return;
  if (connected) {
    backendStatus.textContent = '后端：在线';
    backendStatus.title = endpoint || '';
  } else {
    backendStatus.textContent = '后端：离线（将使用示例答案）';
    backendStatus.title = '请启动后端或设置可用的 /api/ask 地址';
  }
}

/** @typedef {{ role: 'user' | 'assistant', content: string }} ChatMessage */

/** @type {ChatMessage[]} */
let messages = [];
let isSending = false;

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const mockToggle = document.getElementById('mockToggle');
const configBtn = document.getElementById('configBtn');
const backendStatus = document.getElementById('backendStatus');
const roleBtn = document.getElementById('roleBtn');
const loginBtn = document.getElementById('loginBtn');
const loginModal = document.getElementById('loginModal');
const loginClose = document.getElementById('loginClose');
const loginCancel = document.getElementById('loginCancel');
const loginForm = document.getElementById('loginForm');
const loginUsername = document.getElementById('loginUsername');
const loginPassword = document.getElementById('loginPassword');
const loginError = document.getElementById('loginError');
// 登录表单元素
const registerConfirm = document.getElementById('registerConfirm');
const registerRole = document.getElementById('registerRole');
const goRegisterBtn = document.getElementById('goRegister');
const goLoginBtn = document.getElementById('goLogin');
const registerForm = document.getElementById('registerForm');
const registerUsername = document.getElementById('registerUsername');
const registerPassword = document.getElementById('registerPassword');
const registerError = document.getElementById('registerError');
let authMode = 'login';

const TOKEN_KEY = 'qa_auth_token';
const ROLE_KEY = 'qa_user_role'; // 'student' | 'teacher'

init();

function init() {
  // 检查用户角色，如果是教师则跳转到教师界面
  const currentRole = loadRole();
  if (currentRole === 'teacher' && window.location.pathname.includes('index.html')) {
    window.location.href = './teacher.html';
    return;
  }
  
  // 如果当前在教师界面但角色是学生，跳转回学生界面
  if (currentRole === 'student' && window.location.pathname.includes('teacher.html')) {
    window.location.href = './index.html';
    return;
  }

  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    messages = saved ? JSON.parse(saved) : [];
  } catch (e) {
    messages = [];
  }
  renderMessages();

  try {
    const mock = localStorage.getItem(MOCK_KEY);
    mockToggle.checked = mock === '1';
  } catch (e) {}

  sendBtn.addEventListener('click', () => {
    void handleSend();
  });
  clearBtn.addEventListener('click', () => {
    if (messages.length === 0) return;
    const yes = confirm('确定要清空当前对话吗？');
    if (!yes) return;
    messages = [];
    persist();
    renderMessages();
  });
  mockToggle.addEventListener('change', () => {
    localStorage.setItem(MOCK_KEY, mockToggle.checked ? '1' : '0');
  });
  if (configBtn) {
    configBtn.addEventListener('click', async () => {
      try {
        const current = localStorage.getItem('qa_api_endpoint') || API_ENDPOINT || '';
        const next = prompt('请输入后端 /api/ask 地址：', current);
        if (!next) return;
        localStorage.setItem('qa_api_endpoint', next);
        API_ENDPOINT = next;
        await updateBackendHealth();
        alert('已保存。');
      } catch (e) {}
    });
  }
  if (roleBtn) {
    const current = loadRole();
    updateRoleUI(current);
    roleBtn.addEventListener('click', () => {
      const next = toggleRole();
      updateRoleUI(next);
    });
  }
  if (loginBtn) {
    loginBtn.addEventListener('click', () => openLogin());
  }
  if (loginClose) loginClose.addEventListener('click', (e) => { e.preventDefault(); closeLogin(true); });
  if (loginCancel) loginCancel.addEventListener('click', (e) => { e.preventDefault(); closeLogin(true); });
  if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await handleLogin();
    });
  }
  if (registerForm) {
    registerForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await handleRegister();
    });
  }
  if (loginModal) {
    loginModal.addEventListener('click', (e) => {
      if (e.target === loginModal) closeLogin(true);
    });
  }
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeLogin(true);
  });
  if (goRegisterBtn) goRegisterBtn.addEventListener('click', () => { authMode = 'register'; updateAuthModeUI(); });
  if (goLoginBtn) goLoginBtn.addEventListener('click', () => { authMode = 'login'; updateAuthModeUI(); });
  inputEl.addEventListener('input', autoResizeTextarea);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  });

  autoResizeTextarea();
  void updateBackendHealth();
}

function autoResizeTextarea() {
  inputEl.style.height = 'auto';
  const next = Math.min(160, Math.max(40, inputEl.scrollHeight));
  inputEl.style.height = next + 'px';
}

function renderMessages() {
  messagesEl.innerHTML = '';
  for (const msg of messages) {
    const row = document.createElement('div');
    row.className = 'message ' + msg.role;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = msg.role === 'user' ? '我' : '答';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = msg.content;

    row.appendChild(avatar);
    row.appendChild(bubble);
    messagesEl.appendChild(row);
  }
  if (messages.length === 0) {
    const hint = document.createElement('div');
    hint.className = 'hint';
    hint.textContent = '开始提问吧：例如“帮我总结一下这段文字”或“上海今天的天气怎么样？”';
    messagesEl.appendChild(hint);
  }
  scrollToBottom();
}

function scrollToBottom() {
  messagesEl.parentElement.scrollTop = messagesEl.parentElement.scrollHeight;
}

function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  } catch (e) {}
}

async function handleSend() {
  if (isSending) return;
  const text = inputEl.value.trim();
  if (!text) return;

  isSending = true;
  setSendingState(true);

  addMessage({ role: 'user', content: text });
  inputEl.value = '';
  autoResizeTextarea();

  try {
    const answer = await ask(text);
    addMessage({ role: 'assistant', content: answer });
  } catch (err) {
    const reason = err && typeof err === 'object' && 'message' in err ? err.message : String(err);
    addMessage({ role: 'assistant', content: '抱歉，回答失败：' + reason });
  } finally {
    isSending = false;
    setSendingState(false);
  }
}

function setSendingState(sending) {
  sendBtn.disabled = sending;
  messagesEl.setAttribute('aria-busy', String(sending));
  sendBtn.textContent = sending ? '发送中…' : '发送';
}

function addMessage(message) {
  messages.push(message);
  persist();
  renderMessages();
}

async function ask(question) {
  if (mockToggle.checked) {
    await sleep(400);
    return mockAnswer(question);
  }

  try {
    if (isFileMode && (!API_ENDPOINT || API_ENDPOINT === '/api/ask')) {
      await findWorkingEndpoint();
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    const res = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: buildHeaders(),
      body: JSON.stringify({ question }),
      signal: controller.signal
    });
    clearTimeout(timeout);
    if (!res.ok) {
      throw new Error('HTTP ' + res.status);
    }
    const data = await res.json();
    if (data && typeof data.answer === 'string') {
      return data.answer;
    }
    if (data && typeof data.result === 'string') return data.result;
    if (data && typeof data.message === 'string') return data.message;
    return JSON.stringify(data);
  } catch (e) {
    try {
      const prev = API_ENDPOINT;
      await findWorkingEndpoint();
      if (API_ENDPOINT && API_ENDPOINT !== prev) {
        const controller2 = new AbortController();
        const timeout2 = setTimeout(() => controller2.abort(), 15000);
        const res2 = await fetch(API_ENDPOINT, {
          method: 'POST',
          headers: buildHeaders(),
          body: JSON.stringify({ question }),
          signal: controller2.signal
        });
        clearTimeout(timeout2);
        if (res2.ok) {
          const data2 = await res2.json();
          if (data2 && typeof data2.answer === 'string') return data2.answer;
          if (data2 && typeof data2.result === 'string') return data2.result;
          if (data2 && typeof data2.message === 'string') return data2.message;
          return JSON.stringify(data2);
        }
      }
    } catch (_) {}

    const err = e && typeof e === 'object' && 'message' in e ? e.message : String(e);
    console.warn('请求失败，使用示例答案兜底：', err);
    return mockAnswer(question) + '\n\n（已使用本地示例回答；请启动后端或检查端口/CORS 配置）';
  }
}

function mockAnswer(q) {
  const trimmed = q.trim();
  if (trimmed.includes('天气')) {
    return '我是示例回答：今天天气晴到多云，气温 18-26℃，出门记得带太阳镜～';
  }
  if (trimmed.length < 30) {
    return '这是示例回答，用于展示界面交互。你的问题是：“' + trimmed + '”。';
  }
  return '这是示例回答:\n' + summarize(trimmed);
}

function summarize(text) {
  const max = 120;
  if (text.length <= max) return text;
  return text.slice(0, max) + '…';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function openLogin() {
  if (!loginModal) return;
  loginError.hidden = true;
  loginModal.hidden = false;
  loginUsername.focus();
}

function closeLogin(reset) {
  if (!loginModal) return;
  loginModal.hidden = true;
  if (reset && loginForm) loginForm.reset();
  if (loginError) loginError.hidden = true;
  authMode = 'login';
  updateAuthModeUI();
}

function buildHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) headers['Authorization'] = 'Bearer ' + token;
  const role = loadRole();
  if (role) headers['X-User-Role'] = role;
  return headers;
}

async function handleLogin() {
  const username = loginUsername.value.trim();
  const password = loginPassword.value.trim();
  if (!username || !password) {
    showLoginError('请输入用户名和密码');
    return;
  }

  try {
    const maybeAuthEndpoint = (API_ENDPOINT || '').replace('/api/ask', '/api/login');
    let token = null;
    try {
      const res = await fetch(maybeAuthEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      if (res.ok) {
        const data = await res.json();
        token = data && (data.token || data.access_token);
      }
    } catch (_) {}

    if (!token) {
      token = btoa(username + ':' + password);
    }
    localStorage.setItem(TOKEN_KEY, token);
    closeLogin(true);
    alert('登录成功');
  } catch (e) {
    showLoginError('登录失败，请重试');
  }
}

async function handleRegister() {
  const username = (registerUsername ? registerUsername.value : '').trim();
  const password = (registerPassword ? registerPassword.value : '').trim();
  const confirm = registerConfirm ? registerConfirm.value.trim() : '';
  const chosenRole = registerRole ? registerRole.value : 'student';
  if (!username || !password || !confirm) {
    showRegisterError('请完整填写用户名、密码和确认密码');
    return;
  }
  if (password !== confirm) {
    showRegisterError('两次输入的密码不一致');
    return;
  }
  try {
    const maybeAuthEndpoint = (API_ENDPOINT || '').replace('/api/ask', '/api/register');
    let ok = false;
    try {
      const res = await fetch(maybeAuthEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      ok = res.ok;
    } catch (_) {}

    // 无后端时，直接当作注册成功并登录（生成伪 token）
    if (ok || true) {
      const token = btoa(username + ':' + password);
      localStorage.setItem(TOKEN_KEY, token);
      // 保存选择的身份，并根据角色跳转
      localStorage.setItem(ROLE_KEY, chosenRole === 'teacher' ? 'teacher' : 'student');
      closeLogin(true);
      alert('注册并登录成功');
    }
  } catch (e) {
    showRegisterError('注册失败，请重试');
  }
}

function updateAuthModeUI() {
  const loginFormEl = document.getElementById('loginForm');
  const registerFormEl = document.getElementById('registerForm');
  if (!loginFormEl || !registerFormEl) return;
  if (authMode === 'login') {
    document.getElementById('loginTitle').textContent = '登录';
    loginFormEl.hidden = false;
    registerFormEl.hidden = true;
  } else {
    document.getElementById('loginTitle').textContent = '注册';
    loginFormEl.hidden = true;
    registerFormEl.hidden = false;
  }
}

function showRegisterError(msg) {
  if (!registerError) return;
  registerError.textContent = msg;
  registerError.hidden = false;
}

function showLoginError(msg) {
  if (!loginError) return;
  loginError.textContent = msg;
  loginError.hidden = false;
}

function loadRole() {
  const r = localStorage.getItem(ROLE_KEY);
  return r === 'teacher' ? 'teacher' : 'student';
}

function toggleRole() {
  const current = loadRole();
  const next = current === 'student' ? 'teacher' : 'student';
  localStorage.setItem(ROLE_KEY, next);
  
  // 根据新角色跳转页面
  if (next === 'teacher') {
    window.location.href = './teacher.html';
  } else {
    window.location.href = './index.html';
  }
  
  return next;
}

function updateRoleUI(role) {
  const roleBtn = document.getElementById('roleBtn');
  if (!roleBtn) return;
  roleBtn.textContent = '身份：' + (role === 'teacher' ? '老师' : '学生');
  document.body.dataset.role = role;
}
