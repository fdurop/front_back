import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

files_bp = Blueprint("files", __name__)

def safe_filename(filename: str) -> str:
    """
    保留原始文件名，但去掉路径分隔符，防止目录穿越
    """
    return filename.replace("/", "").replace("\\", "")

@files_bp.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "未上传文件"}), 400
    file = request.files["file"]
    filename = safe_filename(file.filename)  # 保留原始名字
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    return jsonify({"success": True, "filename": filename})

@files_bp.route("/files", methods=["GET"])
def list_files():
    files = []
    for f in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, f)
        if os.path.isfile(path):
            files.append({"filename": f, "size": os.path.getsize(path)})
    return jsonify({"success": True, "files": files})

@files_bp.route("/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    filename = safe_filename(filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "文件不存在"}), 404
