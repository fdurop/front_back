#!/usr/bin/env python3
"""
启动后端服务器的脚本
"""
import os
import sys
import subprocess
import time
import requests

def check_port_available(port):
    """检查端口是否可用"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result != 0

def start_server():
    """启动后端服务器"""
    print("�� 正在启动 fdurop 智能问答助手后端服务器...")
    
    # 检查端口是否被占用
    port = 8000
    if not check_port_available(port):
        print(f"❌ 端口 {port} 已被占用，请先关闭占用该端口的程序")
        return False
    
    # 检查依赖
    try:
        import flask
        import werkzeug
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install flask werkzeug")
        return False
    
    # 启动服务器
    try:
        print(f"📁 文件上传目录: {os.path.abspath('src')}")
        print(f"�� 服务器地址: http://127.0.0.1:{port}")
        print("⏳ 正在启动服务器...")
        
        # 启动服务器进程
        process = subprocess.Popen([
            sys.executable, 'server.py', 
            '--host', '127.0.0.1', 
            '--port', str(port)
        ])
        
        # 等待服务器启动
        print("⏳ 等待服务器启动...")
        time.sleep(3)
        
        # 测试服务器是否启动成功
        try:
            response = requests.get(f'http://127.0.0.1:{port}/health', timeout=5)
            if response.status_code == 200:
                print("✅ 服务器启动成功！")
                print(f"�� 前端地址: http://127.0.0.1:{port}")
                print(f"📚 教师界面: file://{os.path.abspath('frontend/teacher.html')}")
                print("�� 提示: 保持此窗口打开，服务器将持续运行")
                print("🔄 按 Ctrl+C 停止服务器")
                
                # 保持进程运行
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 正在停止服务器...")
                    process.terminate()
                    process.wait()
                    print("✅ 服务器已停止")
                
                return True
            else:
                print(f"❌ 服务器响应异常: {response.status_code}")
                process.terminate()
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 无法连接到服务器: {e}")
            process.terminate()
            return False
            
    except Exception as e:
        print(f"❌ 启动服务器失败: {e}")
        return False

if __name__ == '__main__':
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n👋 再见！")
