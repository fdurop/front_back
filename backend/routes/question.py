# routes/qa.py
from flask import Blueprint, request, jsonify
from models.llm_api import call_ollama,extract_after_think

qa_bp = Blueprint("qa", __name__)

@qa_bp.route("/", methods=["POST"])
def qa():
    data = request.get_json()
    print("接收到请求:", data)  # 查看前端是否传数据
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "缺少问题参数"}), 400

    # 调用大模型
    answer = call_ollama(question)
    # 清理掉 <think> 部分
    clean_answer = extract_after_think(answer)

    return jsonify({"answer": clean_answer})