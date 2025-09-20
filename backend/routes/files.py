import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from models.process_folder import build_multimodal_knowledge_graph
from pathlib import Path

UPLOAD_FOLDER = "input"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
BASE_DIR = Path(__file__).resolve().parent.parent  # 获取根目录
input_file_path = BASE_DIR / "input"
output_file_path = BASE_DIR / "output"


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

    result = build_multimodal_knowledge_graph(
        neo4j_uri="bolt://101.132.130.25:7687",
        neo4j_user="neo4j",
        neo4j_password="wangshuxvan@1",
        deepseek_api_key="sk-c28ec338b39e4552b9e6bded47466442",
        input_dir = input_file_path,
        output_dir = output_file_path,
        document_name="Arduino课程PPT",
        fast_mode=False,
        clear_database=True,
        verbose=True
    )

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
