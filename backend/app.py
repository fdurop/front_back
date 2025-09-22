# app.py
from flask import Flask, jsonify,g
from flask_cors import CORS
from routes.question import qa_bp
from routes.files import files_bp
import os
import uuid
from models.logger import logger

app = Flask(__name__)
CORS(app)

app.register_blueprint(qa_bp, url_prefix="/api/qa")
# 注册文件管理蓝图
app.register_blueprint(files_bp, url_prefix="/api")

@app.before_request
def assign_request_id():
    g.request_id = str(uuid.uuid4())

# 健康检查
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=True)
