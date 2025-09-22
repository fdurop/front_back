前端
这是 **fdurop 智能问答助手** 的前端部分，包含学生端与教师端两个界面，支持在线问答、身份切换、文件上传等功能。

## 📂 项目结构

```

frontend/
├── templates/              # Django 模板文件夹
│   ├── index.html          # 学生界面
│   ├── teacher.html        # 教师界面
│   └── base.html           # 公共布局（可选）
│
├── static/                 # 静态资源
│   ├── css/
│   │   ├── style.css       # 学生界面样式
│   │   ├── teacher.css     # 教师界面样式
│   │   └── common.css      # 公共样式（可选）
│   │
│   └── js/
│       ├── chat.js         # 学生界面逻辑（消息发送、显示）
│       ├── qa.js           # 回答渲染优化（支持段落、公式）
│       ├── teacher.js      # 教师上传逻辑
│       └── header-loader.js# 公共 header 逻辑（如身份切换）
│
└── README.md               # 当前文档

````

## 🚀 功能说明

### 学生界面（`index.html`）
- **问答输入**：支持输入问题并发送。
- **身份切换**：可在“学生 / 教师”间切换。
- **清空对话**：清空当前消息。
- **数学公式支持**：集成 MathJax，支持 `$...$` 行内和 `$$...$$` 块级公式。
- **安全渲染**：集成 DOMPurify 防止 XSS。

### 教师界面（`teacher.html`）
- **身份切换**：一键切换回学生界面。
- **文件上传**：
  - 支持 **选择文件** 和 **拖拽上传**。
  - 文件类型支持：`PDF, Word, PPT, 图片, TXT`。
  - 文件列表展示：已上传的文件显示在列表中。
  - 上传进度条：显示上传进度（支持模拟和真实上传）。
- **清空文件列表**：快速清理已选文件。

### 公共功能
- **角色管理**：身份信息存储在 `localStorage`，跨页面保持一致。
- **后端 API 地址管理**（学生界面可配置）。
- **响应式设计**：在桌面和移动端均可使用。

## ⚙️ 使用方法

### 1. 克隆仓库
```bash
git clone https://github.com/your-repo/fdurop-frontend.git
cd fdurop-frontend
````

### 2. 配置 Django 静态文件

在 `settings.py` 中确认：

```python
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / "frontend/static"]
TEMPLATES[0]['DIRS'] = [BASE_DIR / "frontend/templates"]
```

### 3. 启动 Django

```bash
python manage.py runserver
```

然后访问：

* 学生界面：[http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* 教师界面：[http://127.0.0.1:8000/teacher/](http://127.0.0.1:8000/teacher/)

### 4. 前端构建依赖（可选）

如果需要进一步优化前端，可以使用 npm/yarn 管理依赖：

```bash
npm init -y
npm install dompurify
npm install mathjax
```

然后在 HTML 中引入对应的依赖。

## 🧑‍💻 开发指南

* **添加新功能**：

  1. 在 `templates/` 中新增页面。
  2. 在 `static/css/` 中添加样式文件。
  3. 在 `static/js/` 中编写交互逻辑。

* **对接后端上传接口**：
  修改 `teacher.js` 中的上传逻辑：

  ```javascript
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));

  fetch('/api/upload', {
    method: 'POST',
    body: formData
  })
  .then(res => res.json())
  .then(data => console.log('上传成功', data))
  .catch(err => console.error('上传失败', err));
  ```

* **数学公式渲染**：
  学生端的 `qa.js` 在接收消息后会触发：

  ```javascript
  document.dispatchEvent(new Event("newMessage"));
  ```

  然后 `MathJax.typesetPromise()` 会进行公式渲染。

---

## 📌 待办（TODO）

* [ ] 对接后端文件上传接口。
* [ ] 优化上传文件类型校验。
* [ ] 消息持久化（刷新后保留历史记录）。
* [ ] 更丰富的教师界面（文件管理 / 题库管理）。

---


后端
## 🔧 项目简介

本项目是一个基于 **Flask + Python** 的后端服务，主要功能包括：

* 文件上传与解析（支持 `pdf`、`pptx` 等）
* 调用 **大语言模型 (Ollama)** 进行问答
* 解析结果存储至 **Neo4j 图数据库**
* 支持 **异步后台任务**，上传接口立即返回结果，耗时任务在后台执行
* 内置 **统一日志系统**，支持 `request_id` 追踪单次请求全过程

---

## 📂 项目结构

```
backend/
│── app.py                 # Flask 主入口
│── requirements.txt       # 项目依赖
│
├── models/                # 模型与工具
│   ├── llm_api.py         # 调用 Ollama 模型的封装
│   ├── logger.py          # 日志配置（文件 + 控制台，带 request_id）
│
├── routes/                # 路由模块
│   ├── files.py           # 文件上传与后台任务
│   ├── qa.py              # QA 问答接口
│
├── utils/                 # 工具类
│   ├── file_processor.py  # 文件解析与多模态处理
│   ├── neo4j_writer.py    # 写入 Neo4j 的封装
│
└── uploads/               # 上传文件存储目录
```

---

## ⚙️ 环境依赖

推荐使用 **Python 3.9+**
安装依赖：

```bash
pip install -r requirements.txt
```

### requirements.txt 示例

```
flask
flask-cors
neo4j
requests
```

如果需要 PDF/PPTX 解析，还需：

```
pymupdf
python-pptx
```

---

## 🚀 启动方式

```bash
# 开发模式
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000

# 或者直接运行
python app.py
```

启动后，服务默认运行在：

```
http://127.0.0.1:5000
```

---

## 📝 日志系统

日志配置在 `models/logger.py` 中，包含：

* 控制台输出（INFO 级别以上）
* 文件输出（按日期切分 logs/app.log）
* 日志格式统一为：

  ```
  2025-09-21 18:12:45 [INFO] [request_id=abc123] 文件上传成功: test.pdf
  ```

每个请求会自动生成一个 **request\_id**，贯穿整个处理过程。
这样可以在日志中快速追踪一次请求的全过程。

---

## 📤 文件上传接口

### 请求

```
POST /files/upload
```

### 参数

* form-data:

  * `file`: 上传文件 (pdf/pptx)

### 响应

```json
{
  "success": true,
  "filename": "example.pdf",
  "request_id": "c7f8e2f1-8b3a-4f92-9e21-6c2af61a93e1"
}
```

### 说明

* 文件会存储到 `uploads/` 目录
* 文件解析 & Neo4j 写入会在 **后台线程** 执行，不阻塞接口返回
* 日志中可以通过 `request_id` 追踪整个任务

---

## 💬 QA 问答接口

### 请求

```
POST /qa/
```

### 参数

```json
{
  "question": "什么是Arduino？"
}
```

### 响应

```json
{
  "answer": "Arduino 是一个开源电子原型平台...",
  "request_id": "e2f7a8d1-9b34-412b-b82b-4f59f7f50b2f"
}
```

---

## ⚡ 异步后台任务

### 为什么要后台？

* 文件处理和写入 Neo4j 都可能耗时（几十秒以上）
* 为了提升响应速度，接口立即返回 JSON，后台线程继续处理

### 实现方式

在 `routes/files.py`：

```python
# 启动后台线程
thread = Thread(target=background_task, args=(filename, g.request_id))
thread.daemon = True
thread.start()
```

后台线程会执行：

1. `process_multimodal_files()` 文件解析
2. `output_to_neo4j()` 写入图数据库

所有日志都会带上 `request_id`，便于排查问题。

---

## 🗄️ Neo4j 配置

在 `utils/neo4j_writer.py` 中配置：

```python
neo4j_uri = "bolt://<host>:7687"
neo4j_user = "neo4j"
neo4j_password = "<password>"
```

可在 Neo4j 浏览器访问：

```
http://<host>:7474
```

---

## 🧩 大模型调用 (Ollama)

在 `models/llm_api.py` 中，封装了对 **Ollama API** 的调用：

```python
def call_ollama(prompt: str) -> str:
    url = "http://<ollama_server>:11434/api/generate"
    payload = {"model": "llama2", "prompt": prompt}
    response = requests.post(url, json=payload, stream=True)
    return response.text
```

你需要在 **远程服务器** 部署 Ollama，确保 `11434` 端口可访问。
示例：

```bash
ollama run llama2
```

---

## ✅ TODO

* [ ] 增加任务状态查询接口
* [ ] 文件处理结果持久化到数据库
* [ ] 支持更多文件类型（Word、Excel）
* [ ] 增加 Celery 分布式任务队列（替代线程方式）

---

