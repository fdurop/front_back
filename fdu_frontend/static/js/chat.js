// qa.js — 支持 MathJax 渲染、DOMPurify、安全清理 & 段落换行优化
(function () {
  const sendBtn = document.getElementById("sendBtn");
  const inputBox = document.getElementById("input");
  const messages = document.getElementById("messages");
  const mockToggle = document.getElementById("mockToggle");

  // HTML 实体解码
  function decodeEntities(str) {
    if (!str) return str;
    const doc = new DOMParser().parseFromString(str, "text/html");
    return doc.documentElement.textContent;
  }

  // 格式化文本：段落 <p> 与换行 <br>
  function formatText(text) {
    if (!text) return "";
    // 两个及以上换行 -> </p><p>
    let html = text.replace(/\n{2,}/g, "</p><p>");
    // 单个换行 -> <br>
    html = html.replace(/\n/g, "<br>");
    return `<p>${html}</p>`;
  }

  // 渲染 MathJax
  function renderMath(element) {
    if (!element) return;
    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      MathJax.typesetPromise([element]).catch((err) => {
        console.error("MathJax typesetPromise error:", err);
      });
      return;
    }
    if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
      window.MathJax.startup.promise.then(() => {
        if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
          MathJax.typesetPromise([element]).catch(err =>
            console.error("MathJax typesetPromise error after startup:", err)
          );
        }
      }).catch(err => console.error("MathJax startup.promise rejected:", err));
      return;
    }
    const to = setInterval(() => {
      if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
        clearInterval(to);
        MathJax.typesetPromise([element]).catch(err => console.error("MathJax typesetPromise error:", err));
      }
    }, 120);
  }

  // 安全清理并格式化文本
  function sanitizeAndFormat(rawText) {
    let decoded = decodeEntities(rawText);
    if (window.DOMPurify) {
      decoded = DOMPurify.sanitize(decoded, {ALLOWED_TAGS: [], ALLOWED_ATTR: []});
    } else {
      const tmp = document.createElement('div');
      tmp.textContent = decoded;
      decoded = tmp.innerHTML;
    }
    return formatText(decoded);
  }

  // 添加消息
  function appendMessage(rawText, sender = "user") {
    const div = document.createElement("div");
    div.classList.add("message", sender);
    div.innerHTML = sanitizeAndFormat(rawText);
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    renderMath(div);
    document.dispatchEvent(new Event("newMessage"));
    return div;
  }

  // 更新已存在消息
  function updateMessage(div, rawText) {
    if (!div) return;
    div.innerHTML = sanitizeAndFormat(rawText);
    renderMath(div);
    document.dispatchEvent(new Event("newMessage"));
  }

  // 发送问题
  async function sendQuestion() {
    const question = inputBox.value.trim();
    if (!question) return;

    appendMessage(question, "user");
    inputBox.value = "";

    const lastMsg = appendMessage("正在思考中…", "ai");

    if (mockToggle?.checked) {
      updateMessage(lastMsg, "这是示例答案");
      return;
    }

    const BACKEND_URL =
      localStorage.getItem("qa_api_endpoint") ||
      "http://127.0.0.1:5000/api/qa/";

    try {
      const res = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const errText = await res.text().catch(()=>null);
        updateMessage(lastMsg, `后端错误: ${res.status}` + (errText ? ` — ${errText}` : ""));
        return;
      }

      const data = await res.json().catch(async () => {
        const txt = await res.text().catch(()=>null);
        return { answer: txt || null };
      });

      updateMessage(lastMsg, data.answer || "未返回答案");
    } catch (err) {
      updateMessage(lastMsg, "调用后端失败");
      console.error("调用后端失败:", err);
    }
  }

  function initEvents() {
    sendBtn?.addEventListener("click", sendQuestion);
    inputBox?.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
      }
    });
  }

  window.addEventListener("load", initEvents);
})();
