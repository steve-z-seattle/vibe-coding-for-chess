# 国际象棋 - FastAPI 后端版

一个基于 FastAPI 后端的国际象棋网页游戏，支持人机对战（AI 使用 Minimax + Alpha-Beta 剪枝算法）。

## 项目结构

```
.
├── backend/                  # FastAPI 后端
│   ├── main.py              # FastAPI 主应用
│   ├── chess_game.py        # 国际象棋游戏逻辑
│   ├── ai.py                # AI 算法
│   └── models.py            # Pydantic 数据模型
├── frontend/                # 前端文件
│   └── static/
│       ├── index.html       # 前端页面
│       └── pieces/          # 棋子 SVG 图标
├── pieces/                  # 原始棋子图标（备用）
├── requirements.txt         # Python 依赖
├── run.sh                   # Linux/Mac 启动脚本
└── run.bat                  # Windows 启动脚本
```

## 快速开始

### 1. 安装依赖

确保已安装 Python 3.8+，然后运行启动脚本：

**Linux/Mac:**
```bash
./run.sh
```

**Windows:**
```cmd
run.bat
```

或者手动安装：
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate.bat  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动服务器
cd backend
uvicorn main:app --reload
```

### 2. 访问游戏

- **游戏界面**: http://localhost:8000/static/index.html
- **API 文档**: http://localhost:8000/docs
- **API 根路径**: http://localhost:8000/

## API 接口

### 游戏状态
- `GET /api/game/{game_id}/state` - 获取当前游戏状态

### 游戏操作
- `POST /api/game/{game_id}/reset` - 重置游戏
- `POST /api/game/{game_id}/move` - 执行移动
- `POST /api/game/{game_id}/undo` - 悔棋
- `POST /api/game/{game_id}/ai-move` - AI 走棋

### 查询
- `GET /api/game/{game_id}/valid-moves?row={row}&col={col}` - 获取指定棋子的有效移动
- `GET /api/game/{game_id}/check` - 检查游戏是否结束

## 功能特性

- ✅ 完整的国际象棋规则
  - 所有棋子的标准移动规则
  - 王车易位（Castling）
  - 吃过路兵（En Passant）
  - 兵的升变（Promotion）
  - 将军和将死检测
  - 逼和（Stalemate）检测
  - 材料不足和棋

- ✅ AI 对战
  - Minimax 算法
  - Alpha-Beta 剪枝优化
  - 位置评估函数
  - 自动升变为皇后

- ✅ 前端功能
  - 响应式设计
  - 棋子拖拽/点击移动
  - 有效移动提示
  - 走棋记录
  - 被吃棋子显示
  - 悔棋功能
  - 棋盘翻转

## 技术栈

- **后端**: FastAPI, Python 3.8+
- **前端**: HTML5, CSS3, JavaScript (Vanilla)
- **通信**: RESTful API (HTTP)

## AI 算法

AI 使用 Minimax 算法配合 Alpha-Beta 剪枝，搜索深度默认为 3 层。

评估函数考虑：
- 棋子基础价值
- 棋子位置价值（使用标准开局/中局位置表）
- 行动力（可移动格子数）
- 将军奖励

## 许可证

MIT License
