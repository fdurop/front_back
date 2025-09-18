window.headerLoader = (function() {
  const elements = {};

  function load() {
    elements.roleBtn = document.getElementById('roleBtn');

    // 切换身份按钮
    elements.roleBtn?.addEventListener('click', () => {
      localStorage.setItem('qa_user_role', 'student');
      window.location.href = '/';
    });

    // 设置按钮文本
    if (elements.roleBtn) elements.roleBtn.textContent = '切换到学生身份';

    return Promise.resolve();
  }

  function waitForLoad() {
    return Promise.resolve();
  }

  return { load, waitForLoad };
})();
