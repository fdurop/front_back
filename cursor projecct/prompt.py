from flask import Flask, request, jsonify
from localtry import ask_ollama, extract_final_answer

"""极简问答服务：直接将用户问题转发给本地 Ollama 并返回最终答案。"""


def answer_question(query: str) -> str:
    """直接调用 Ollama，返回提炼后的最终答案。"""
    raw_text = ask_ollama(query)
    return extract_final_answer(raw_text)


# ---------------------- 7. Web 服务（/api/ask） ----------------------
app = Flask(__name__)


# 轻量 CORS 处理（避免额外依赖）
@app.after_request
def add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    resp.headers['Access-Control-Allow-Methods'] = 'POST,OPTIONS'
    return resp


@app.route('/api/ask', methods=['POST', 'OPTIONS'])
def api_ask():
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or data.get('q') or '').strip()
    if not question:
        return jsonify({"error": "missing field: question"}), 400
    try:
        answer = answer_question(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------- 8. 启动方式 ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ollama Q&A Service")
    parser.add_argument('--cli', action='store_true', help='以命令行方式提问并输出答案')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    if args.cli:
        query = input("请输入你的问题：")
        print(answer_question(query))
    else:
        app.run(host=args.host, port=args.port, debug=True)
