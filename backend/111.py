import os
import sys
import traceback


def debug_full_pipeline():
    """完整调试处理流程"""

    # 1. 检查路径
    input_dir = "C:/Users/86131/Desktop/fdurop/backend/input/"
    output_dir = "C:/Users/86131/Desktop/fdurop/backend/output/"

    print("=" * 50)
    print("🔍 调试多模态知识图谱构建流程")
    print("=" * 50)

    # 2. 检查输入目录
    print(f"\n📥 检查输入目录: {input_dir}")
    if not os.path.exists(input_dir):
        print("❌ 输入目录不存在！")
        return False

    files = os.listdir(input_dir)
    doc_files = [f for f in files if f.endswith(('.ppt', '.pptx', '.pdf'))]
    print(f"📄 找到 {len(doc_files)} 个文档文件:")
    for file in doc_files:
        file_path = os.path.join(input_dir, file)
        size = os.path.getsize(file_path) / 1024  # KB
        print(f"   - {file} ({size:.1f} KB)")

    if not doc_files:
        print("❌ 没有找到可处理的文档文件！")
        return False

    # 3. 检查输出目录
    print(f"\n📤 检查输出目录: {output_dir}")
    if not os.path.exists(output_dir):
        print("📁 创建输出目录...")
        os.makedirs(output_dir, exist_ok=True)

    # 4. 检查依赖库
    print(f"\n📚 检查依赖库:")
    try:
        from pptx import Presentation
        print("✅ python-pptx")
    except ImportError:
        print("❌ python-pptx 未安装")
        return False

    try:
        import fitz
        print("✅ PyMuPDF")
    except ImportError:
        print("❌ PyMuPDF 未安装")
        return False

    # 5. 测试文档处理
    print(f"\n🔄 测试文档处理:")
    try:
        # 导入你的处理器
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        from .process_folder import DocumentProcessor

        processor = DocumentProcessor(output_dir)
        print("✅ 文档处理器初始化成功")

        # 处理第一个文件
        test_file = doc_files[0]
        test_path = os.path.join(input_dir, test_file)
        print(f"🧪 测试处理文件: {test_file}")

        if test_file.endswith('.pptx'):
            result = processor.process_pptx(test_path)
            print(f"📊 处理结果: {result}")

        # 检查输出
        print(f"\n📋 检查输出结果:")
        if os.path.exists(output_dir):
            output_files = os.listdir(output_dir)
            print(f"📄 输出文件数量: {len(output_files)}")
            for file in output_files:
                print(f"   - {file}")

            # 检查子目录
            subdirs = ['text', 'images', 'tables', 'formulas', 'code']
            for subdir in subdirs:
                subdir_path = os.path.join(output_dir, subdir)
                if os.path.exists(subdir_path):
                    sub_files = os.listdir(subdir_path)
                    print(f"📁 {subdir}/: {len(sub_files)} 个文件")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_full_pipeline()
    if success:
        print("\n✅ 调试完成，可以继续运行完整流程")
    else:
        print("\n❌ 发现问题，请先解决上述问题")