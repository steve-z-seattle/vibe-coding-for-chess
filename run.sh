#!/bin/bash

# 国际象棋 FastAPI 后端启动脚本

echo "=================================="
echo "  Chess API - FastAPI Backend"
echo "=================================="
echo ""

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "正在创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "正在激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "正在安装依赖..."
pip install -q -r requirements.txt

# 启动服务器
echo ""
echo "启动 FastAPI 服务器..."
echo "API 文档地址: http://localhost:8000/docs"
echo "前端访问地址: http://localhost:8000/static/index.html"
echo ""

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
