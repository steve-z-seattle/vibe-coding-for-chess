# 国际象棋 - FastAPI 后端版

一个基于 FastAPI 后端的国际象棋网页游戏，支持人机对战（AI 使用 Minimax + Alpha-Beta 剪枝算法）。

## 项目结构

```
.
├── backend/                  # FastAPI 后端
│   ├── main.py              # FastAPI 主应用
│   ├── chess_game.py        # 国际象棋游戏逻辑
│   ├── ai.py                # AI 算法
│   ├── models.py            # Pydantic 数据模型
│   └── tests/               # 单元测试
│       ├── test_chess_game.py   # 游戏逻辑测试
│       ├── test_ai.py           # AI 算法测试
│       ├── test_api.py          # API 集成测试
│       └── conftest.py          # 测试配置
├── frontend/                # 前端文件
│   └── static/
│       ├── index.html       # 前端页面
│       └── pieces/          # 棋子 SVG 图标
├── pieces/                  # 原始棋子图标（备用）
├── requirements.txt         # Python 依赖
├── run.sh                   # Linux/Mac 启动脚本
├── run.bat                  # Windows 启动脚本
├── run_tests.sh             # Linux/Mac 测试脚本
└── run_tests.bat            # Windows 测试脚本
```

## 快速开始

### 1. 安装依赖

确保已安装 Python 3.12+，然后运行启动脚本：

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
# 创建虚拟环境（确保使用 Python 3.12+）
python3.12 -m venv venv

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
- `POST /api/game/{game_id}/ai-move` - AI 走棋

### 查询
- `GET /api/game/{game_id}/valid-moves?row={row}&col={col}` - 获取指定棋子的有效移动
- `GET /api/game/{game_id}/check` - 检查游戏是否结束
- `GET /api/game/{game_id}/history/{move_number}` - 获取指定步数的历史局面（用于棋局回顾）

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
  - 三种难度级别（简单/中等/困难）
  - 快速响应（简单模式 < 1秒）

- ✅ 前端功能
  - 响应式设计
  - 棋子拖拽/点击移动
  - 有效移动提示
  - 走棋记录
  - 被吃棋子显示（净数量）
  - 棋局回顾导航（支持键盘快捷键）
  - 走棋音效
  - 棋盘翻转

## 技术栈

- **后端**: FastAPI, Python 3.12+
- **前端**: HTML5, CSS3, JavaScript (Vanilla)
- **通信**: RESTful API (HTTP)

## AI 算法

AI 使用 Minimax 算法配合 Alpha-Beta 剪枝，支持三种难度级别：

| 难度 | 搜索深度 | 思考时间 |
|------|----------|----------|
| 简单 | 1-2 层 | ~0.5 秒 |
| 中等 | 2-3 层 | ~1 秒 |
| 困难 | 3-4 层 | ~1.5 秒 |

评估函数考虑：
- 棋子基础价值
- 棋子位置价值（使用标准开局/中局位置表）
- 将军奖励
- Zobrist 哈希（用于置换表优化）

## 单元测试

项目包含完整的单元测试套件，使用 pytest 框架。

### 运行测试

**使用脚本（推荐）：**
```bash
# Linux/Mac
./run_tests.sh

# Windows
run_tests.bat
```

**手动运行：**
```bash
cd backend
python -m pytest tests/ -v
```

**带覆盖率报告：**
```bash
cd backend
python -m pytest tests/ --cov=. --cov-report=html
```

### 测试覆盖

- **test_chess_game.py** (50+ 测试): 游戏逻辑测试
  - 棋盘初始化
  - 各棋子移动规则（兵、马、象、车、后、王）
  - 特殊规则（王车易位、吃过路兵、升变）
  - 将军、将死、逼和检测
  
- **test_ai.py** (35+ 测试): AI 算法测试
  - 棋子价值评估
  - 位置评估表
  - 最佳走法生成
  - 置换表功能
  - 快速走法操作
  - 时间管理
  
- **test_api.py** (35+ 测试): API 集成测试
  - 所有 REST API 端点
  - 游戏状态获取
  - 走棋操作
  - 游戏重置
  - 多人游戏隔离
  - 错误处理

## 许可证

MIT License
