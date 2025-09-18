// qa.js
(function () {
  const sendBtn = document.getElementById("sendBtn");
  const inputBox = document.getElementById("input");
  const messages = document.getElementById("messages");
  const mockToggle = document.getElementById("mockToggle");

  async function sendQuestion() {
    const question = inputBox.value.trim();
    if (!question) return;

    appendMessage(question, "user");
    inputBox.value = "";

    const lastMsg = appendMessage("正在思考中…", "ai");

    if (mockToggle?.checked) {
      lastMsg.textContent = "这是示例答案";
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
      const data = await res.json();
      lastMsg.textContent = data.answer || "未返回答案";
    } catch (err) {
      lastMsg.textContent = "调用后端失败";
      console.error(err);
    }
  }

  function appendMessage(text, sender) {
    const div = document.createElement("div");
    div.classList.add("message", sender);
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
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

  // 页面加载时运行
  window.addEventListener("load", initEvents);
})();
