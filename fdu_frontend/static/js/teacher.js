(function () {
  // 修改端口匹配后端 Flask 的 5000
  const API_BASE = localStorage.getItem("qa_api_base") || "http://127.0.0.1:5000";

  // 强制身份为 teacher
  localStorage.setItem("qa_user_role", "teacher");

  async function initHeader() {
    try {
      await window.headerLoader.load();
      await window.headerLoader.waitForLoad();
      console.log("[teacher.js] Header 初始化完成");
    } catch (err) {
      console.error("Header 初始化失败:", err);
    }
  }

  function initFileUpload() {
    const uploadSection = document.getElementById("uploadSection");
    const uploadArea = document.getElementById("uploadArea");
    const fileInput = document.getElementById("fileInput");
    const chooseFileBtn = document.getElementById("chooseFileBtn");
    const fileItems = document.getElementById("fileItems");
    const uploadProgress = document.getElementById("uploadProgress");
    const progressFill = document.getElementById("progressFill");
    const progressText = document.getElementById("progressText");
    const statusMessage = document.getElementById("statusMessage");

    async function checkBackend() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        const backendStatus = document.getElementById("backendStatus");
        backendStatus.textContent = res.ok ? "后端：在线" : "后端：离线";
        uploadSection.hidden = false;
        if (res.ok) loadFileList();
      } catch {
        const backendStatus = document.getElementById("backendStatus");
        backendStatus.textContent = "后端：离线";
        uploadSection.hidden = false;
      }
    }
    checkBackend();

    chooseFileBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", e => handleFiles(e.target.files));
    uploadArea.addEventListener("dragover", e => { e.preventDefault(); uploadArea.classList.add("drag-over"); });
    uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
    uploadArea.addEventListener("drop", e => { e.preventDefault(); uploadArea.classList.remove("drag-over"); handleFiles(e.dataTransfer.files); });

    function handleFiles(files) { [...files].forEach(uploadFile); }

    async function uploadFile(file) {
      uploadProgress.hidden = false;
      progressFill.style.width = "0%";
      progressText.textContent = "准备上传...";
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch(`${API_BASE}/api/upload`, {
          method: "POST",
          headers: { "X-User-Role": "teacher", "Authorization": localStorage.getItem("qa_auth_token") || "" },
          body: formData
        });
        const result = await res.json();
        if (res.ok && result.success) {
          progressFill.style.width = "100%";
          progressText.textContent = "上传完成！";
          showStatus("文件上传成功", "success");
          setTimeout(loadFileList, 1000);
        } else throw new Error(result.error || "上传失败");
      } catch (err) { showStatus(`上传失败: ${err.message}`, "error"); }
      finally { setTimeout(() => uploadProgress.hidden = true, 2000); }
    }

    async function loadFileList() {
      try {
        const res = await fetch(`${API_BASE}/api/files`, { headers: { "X-User-Role": "teacher" } });
        const result = await res.json();
        if (res.ok && result.success) renderFileList(result.files);
        else throw new Error(result.error || "加载失败");
      } catch (err) { showStatus(`获取文件失败: ${err.message}`, "error"); }
    }

    function renderFileList(files) {
      fileItems.innerHTML = files.length ? "" : '<div class="no-files">暂无文件</div>';
      files.forEach(file => {
        const el = document.createElement("div");
        el.className = "file-item";
        el.innerHTML = `
          <div class="file-info">
            <span class="file-icon">${getFileIcon(file.filename)}</span>
            <div>
              <div class="file-name">${file.filename}</div>
              <div class="file-size">${formatSize(file.size)}</div>
            </div>
          </div>
          <button class="file-action-btn" onclick="deleteFile('${file.filename}')">🗑️</button>`;
        fileItems.appendChild(el);
      });
    }

    window.deleteFile = async function (filename) {
      if (!confirm(`确定要删除 "${filename}" 吗？`)) return;
      try {
        const res = await fetch(`${API_BASE}/api/files/${encodeURIComponent(filename)}`, { method: "DELETE", headers: { "X-User-Role": "teacher" } });
        const result = await res.json();
        if (res.ok && result.success) { showStatus("删除成功", "success"); loadFileList(); }
        else throw new Error(result.error || "删除失败");
      } catch (err) { showStatus(`删除失败: ${err.message}`, "error"); }
    };

    function getFileIcon(name) {
      const ext = name.split(".").pop().toLowerCase();
      const icons = { pdf: "📄", doc: "📝", docx: "📝", ppt: "📊", pptx: "📊", jpg: "🖼️", jpeg: "🖼️", png: "🖼️", gif: "🖼️", txt: "📄" };
      return icons[ext] || "📁";
    }

    function formatSize(bytes) {
      if (!bytes) return "0 B";
      const units = ["B","KB","MB","GB"];
      const i = Math.floor(Math.log(bytes)/Math.log(1024));
      return (bytes/Math.pow(1024,i)).toFixed(2) + " " + units[i];
    }

    function showStatus(msg, type="info") {
      statusMessage.textContent = msg;
      statusMessage.className = `status-message ${type}`;
      statusMessage.hidden = false;
      setTimeout(() => statusMessage.hidden = true, 3000);
    }
  }

  window.addEventListener("load", async () => {
    await initHeader();
    initFileUpload();
  });
})();
