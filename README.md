# 语音对话 Agent - 文字原型

这是第一版最小可用实现：命令行文字对话，使用 OpenAI 兼容的 Chat Completions API，并保留多轮上下文。默认配置面向 MiniMax Coding/Token Plan，也保留了 `OPENAI_*` 环境变量兼容。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 配置

复制配置模板：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

```env
MINIMAX_API_KEY=your_minimax_api_key_here
MINIMAX_BASE_URL=https://api.minimaxi.com/v1
MINIMAX_MODEL=MiniMax-M2.7
MINIMAX_API_MODE=chat
MAX_HISTORY_TURNS=20
```

说明：

- `MINIMAX_API_KEY`：你的 MiniMax API Key。
- `MINIMAX_BASE_URL`：MiniMax OpenAI 兼容地址，默认 `https://api.minimaxi.com/v1`。
- `MINIMAX_MODEL`：模型名，MiniMax Coding/Token Plan 常用 `MiniMax-M2.7` 或 `MiniMax-M2.7-highspeed`。
- `MINIMAX_API_MODE`：接口模式，MiniMax 兼容接口使用 `chat`。
- `MAX_HISTORY_TURNS`：保留最近多少轮上下文，默认 `20`。

也可以直接在 PowerShell 里设置：

```powershell
$env:MINIMAX_API_KEY="your_minimax_api_key_here"
$env:MINIMAX_BASE_URL="https://api.minimaxi.com/v1"
$env:MINIMAX_MODEL="MiniMax-M2.7"
$env:MINIMAX_API_MODE="chat"
```

## 运行

```powershell
python main.py
```

可用命令：

- `exit` / `quit` / `退出`：结束程序
- `reset` / `清空`：清空当前对话上下文

## 文件说明

- `main.py`：命令行交互入口
- `agent.py`：OpenAI 兼容接口调用和错误处理
- `conversation.py`：多轮对话上下文管理
- `config.py`：环境变量和默认配置
- `.env.example`：本地配置模板，不要把真实 `.env` 提交到版本控制
