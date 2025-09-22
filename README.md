前端
# 安装依赖
pip install -r requirements.txt

# 数据库初始化
python manage.py migrate

# 启动开发服务器
python manage.py runserver

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

