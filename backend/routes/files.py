import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from models.processor import process_multimodal_files
from models.extractor_connector import output_to_neo4j
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

    # 示例调用
    result = process_multimodal_files(
        input_dir=str(input_file_path),
        output_dir=str(output_file_path),
        clip_model_path=r"F:\Models\clip-vit-base-patch32",  # 可选：本地CLIP模型路径
        fast_mode=False,  # 设置为True可跳过耗时的CLIP描述生成
        file_types=['pdf', 'pptx']  # 支持的文件类型
    )

    # 调用示例
    result = output_to_neo4j(
        output_dir=str(output_file_path),
        deepseek_api_key="sk-c28ec338b39e4552b9e6bded47466442",
        neo4j_uri="bolt://101.132.130.25:7687",
        neo4j_user="neo4j",
        neo4j_password="wangshuxvan@1",
        ppt_name="Arduino课程PPT",
        clear_database=False,  # 是否清空数据库
        show_examples=True  # 是否显示查询示例
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
