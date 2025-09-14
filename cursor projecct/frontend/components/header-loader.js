/**
 * 通用Header加载器
 * 可以在任何页面中使用
 */
class HeaderLoader {
  constructor(containerId = 'headerContainer') {
    this.containerId = containerId;
    this.container = document.getElementById(containerId);
    this.loaded = false;
    
    // 内联header HTML，避免fetch问题
    this.headerHtml = `
      <header class="app-header">
        <div class="title">fdurop——智能问答助手</div>
        <div class="actions">
          <label class="toggle">
            <input type="checkbox" id="mockToggle">
            <span>使用示例答案</span>
          </label>
          <span id="backendStatus" class="status-badge" title="后端连接状态">后端：检查中…</span>
          <button id="roleBtn" class="secondary-btn" title="切换身份">身份：学生</button>
          <button id="configBtn" class="secondary-btn" title="设置后端地址">设置后端</button>
          <button id="loginBtn" class="secondary-btn" title="登录或注册账号">登录&注册</button>
          <button id="clearBtn" class="secondary-btn" title="清空对话">清空</button>
        </div>
      </header>
    `;
  }
  
  async load() {
    if (this.loaded) return;
    
    try {
      // 直接使用内联HTML，避免fetch问题
      this.container.innerHTML = this.headerHtml;
      this.loaded = true;
      
      // 触发header加载完成事件
      this.container.dispatchEvent(new CustomEvent('headerLoaded'));
      
      return true;
    } catch (error) {
      console.error('加载header失败:', error);
      this.showErrorHeader();
      return false;
    }
  }
  
  showErrorHeader() {
    this.container.innerHTML = `
      <header class="app-header">
        <div class="title">fdurop——智能问答助手</div>
        <div class="actions">
          <span class="error-message">Header加载失败</span>
        </div>
      </header>
    `;
  }
  
  // 获取header中的元素
  getElement(id) {
    return this.container.querySelector(`#${id}`);
  }
  
  // 等待header加载完成
  async waitForLoad() {
    if (this.loaded) return;
    
    return new Promise((resolve) => {
      this.container.addEventListener('headerLoaded', resolve, { once: true });
    });
  }
  
  // 设置角色按钮文本
  setRoleText(role) {
    const roleBtn = this.getElement('roleBtn');
    if (roleBtn) {
      roleBtn.textContent = `身份：${role === 'teacher' ? '老师' : '学生'}`;
    }
  }
  
  // 隐藏/显示特定按钮
  toggleButton(buttonId, show) {
    const button = this.getElement(buttonId);
    if (button) {
      button.style.display = show ? '' : 'none';
    }
  }
}

// 全局header加载器实例
window.headerLoader = new HeaderLoader();
