# routes/files.py
import os
import uuid
from flask import Blueprint, request, jsonify, g
from pathlib import Path
from models.processor import process_multimodal_files
from models.extractor_connector import output_to_neo4j
from werkzeug.utils import secure_filename
from models.logger import logger  # 统一 logger

UPLOAD_FOLDER = "input"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
BASE_DIR = Path(__file__).resolve().parent.parent
input_file_path = BASE_DIR / "input"
output_file_path = BASE_DIR / "output"

files_bp = Blueprint("files", __name__)

@files_bp.before_request
def assign_request_id():
    g.request_id = str(uuid.uuid4())

def safe_filename(filename: str) -> str:
    return filename.replace("/", "").replace("\\", "")

@files_bp.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        logger.warning("未上传文件", extra={"request_id": g.request_id})
        return jsonify({"success": False, "error": "未上传文件", "request_id": g.request_id}), 400

    file = request.files["file"]
    filename = safe_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    logger.info(f"文件上传成功: {filename}", extra={"request_id": g.request_id})

    try:
        process_multimodal_files(
            input_dir=str(input_file_path),
            output_dir=str(output_file_path),
            clip_model_path=r"F:\Models\clip-vit-base-patch32",
            fast_mode=False,
            file_types=['pdf', 'pptx']
        )
        logger.info(f"文件处理完成: {filename}", extra={"request_id": g.request_id})
    except Exception as e:
        logger.error(f"文件处理失败: {filename}, 错误: {e}", extra={"request_id": g.request_id})

    try:
        output_to_neo4j(
            output_dir=str(output_file_path),
            deepseek_api_key="sk-c28ec338b39e4552b9e6bded47466442",
            neo4j_uri="bolt://101.132.130.25:7687",
            neo4j_user="neo4j",
            neo4j_password="wangshuxvan@1",
            ppt_name="Arduino课程PPT",
            clear_database=False,
            show_examples=True
        )
        logger.info(f"数据写入Neo4j成功: {filename}", extra={"request_id": g.request_id})
    except Exception as e:
        logger.error(f"写入Neo4j失败: {filename}, 错误: {e}", extra={"request_id": g.request_id})

    return jsonify({"success": True, "filename": filename, "request_id": g.request_id})

@files_bp.route("/files", methods=["GET"])
def list_files():
    files = [{"filename": f, "size": os.path.getsize(os.path.join(UPLOAD_FOLDER, f))}
             for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    logger.info(f"加载文件列表，共 {len(files)} 个文件", extra={"request_id": g.request_id})
    return jsonify({"success": True, "files": files, "request_id": g.request_id})

@files_bp.route("/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    filename = safe_filename(filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"文件删除成功: {filename}", extra={"request_id": g.request_id})
        return jsonify({"success": True, "request_id": g.request_id})
    logger.warning(f"文件不存在: {filename}", extra={"request_id": g.request_id})
    return jsonify({"success": False, "error": "文件不存在", "request_id": g.request_id}), 404
