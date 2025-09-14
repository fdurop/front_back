import os
import sys
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from localtry import ask_ollama, extract_final_answer

app = Flask(__name__)

# 文件上传配置
UPLOAD_FOLDER = 'src'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'ppt', 'pptx'}

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.after_request
def add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-User-Role'
    resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS, GET, DELETE'
    resp.headers['Access-Control-Max-Age'] = '86400'  # 24小时
    return resp

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True}), 200

@app.route('/api/ask', methods=['POST', 'OPTIONS'])
def api_ask():
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or data.get('q') or '').strip()
    if not question:
        return jsonify({"error": "missing field: question"}), 400
    try:
        raw_text = ask_ollama(question)
        answer = extract_final_answer(raw_text)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return ('', 204)
    
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({"error": "没有文件被上传"}), 400
        
        file = request.files['file']
        
        # 检查文件名是否为空
        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400
        
        # 检查文件类型
        if file and allowed_file(file.filename):
            # 安全地保存文件名
            filename = secure_filename(file.filename)
            
            # 生成唯一文件名（避免重名）
            base_name, extension = os.path.splitext(filename)
            counter = 1
            final_filename = filename
            
            while os.path.exists(os.path.join(UPLOAD_FOLDER, final_filename)):
                final_filename = f"{base_name}_{counter}{extension}"
                counter += 1
            
            # 保存文件
            file_path = os.path.join(UPLOAD_FOLDER, final_filename)
            file.save(file_path)
            
            # 获取文件信息
            file_size = os.path.getsize(file_path)
            
            return jsonify({
                "success": True,
                "message": "文件上传成功",
                "filename": final_filename,
                "original_name": file.filename,
                "size": file_size,
                "path": file_path
            }), 200
        else:
            return jsonify({"error": "不支持的文件类型"}), 400
            
    except Exception as e:
        return jsonify({"error": f"上传失败: {str(e)}"}), 500

@app.route('/api/files', methods=['GET', 'OPTIONS'])
def list_files():
    """获取已上传的文件列表"""
    if request.method == 'OPTIONS':
        return ('', 204)
        
    try:
        files = []
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    file_stat = os.stat(file_path)
                    files.append({
                        "filename": filename,
                        "size": file_stat.st_size,
                        "upload_time": file_stat.st_mtime,
                        "path": file_path
                    })
        
        return jsonify({
            "success": True,
            "files": files
        }), 200
    except Exception as e:
        return jsonify({"error": f"获取文件列表失败: {str(e)}"}), 500

@app.route('/api/files/<filename>', methods=['DELETE', 'OPTIONS'])
def delete_file(filename):
    """删除指定文件"""
    if request.method == 'OPTIONS':
        return ('', 204)
        
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "文件不存在"}), 404
        
        if not os.path.isfile(file_path):
            return jsonify({"error": "不是有效文件"}), 400
        
        os.remove(file_path)
        
        return jsonify({
            "success": True,
            "message": "文件删除成功"
        }), 200
    except Exception as e:
        return jsonify({"error": f"删除文件失败: {str(e)}"}), 500

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Ollama Q&A Service with File Upload')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    
    print(f"文件上传目录: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"支持的文件类型: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"服务器启动在: http://{args.host}:{args.port}")
    
    app.run(host=args.host, port=args.port, debug=True)


