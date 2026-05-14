from agent import AgentError, VoiceTextAgent
from config import load_settings


EXIT_COMMANDS = {"exit", "quit", "q", "bye", "退出", "再见"}
RESET_COMMANDS = {"reset", "/reset", "清空", "重置"}


def main() -> int:
    try:
        settings = load_settings()
    except RuntimeError as exc:
        print(f"配置错误：{exc}")
        return 1

    agent = VoiceTextAgent(settings)

    print("文字对话 Agent 已启动。")
    print(f"当前模型：{settings.model}")
    print("输入 exit / quit / 退出 结束；输入 reset / 清空 重置上下文。")

    while True:
        try:
            user_text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0

        if not user_text:
            continue

        normalized = user_text.lower()
        if normalized in EXIT_COMMANDS:
            print("已退出。")
            return 0

        if normalized in RESET_COMMANDS:
            agent.reset()
            print("上下文已清空。")
            continue

        try:
            reply = agent.chat(user_text)
        except AgentError as exc:
            print(f"请求失败：{exc}")
            continue

        print(f"Agent> {reply}")


if __name__ == "__main__":
    raise SystemExit(main())
