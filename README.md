# 语音对话 Agent - 文字原型

这是第一版最小可用实现：命令行文字对话，使用 OpenAI Responses API，并保留多轮上下文。后续可以在这个结构上继续接入语音识别和语音合成。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 配置

方式一：使用 `.env` 文件。

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，填入你的 API Key：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5-mini
```

方式二：直接在 PowerShell 里设置环境变量。

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_MODEL="gpt-5-mini"
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
- `agent.py`：OpenAI 调用和上下文管理
- `config.py`：环境变量和默认配置
- `.env.example`：本地配置模板，不要把真实 `.env` 提交到版本控制
