from agent import AgentError, VoiceTextAgent
from config import load_settings


EXIT_COMMANDS = {"exit", "quit", "q", "bye", "tuichu", "zaijian"}
RESET_COMMANDS = {"reset", "/reset", "clear", "qingkong", "chongzhi"}


def main() -> int:
    try:
        settings = load_settings()
    except RuntimeError as exc:
        print(f"Config error: {exc}")
        return 1

    agent = VoiceTextAgent(settings)

    print("Text conversation agent started.")
    print(f"Provider: {settings.provider}")
    print(f"Model: {settings.model}")
    print(f"API mode: {settings.api_mode}")
    if settings.base_url:
        print(f"Base URL: {settings.base_url}")
    print("Type exit/quit to stop, or reset/clear to clear the conversation.")

    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExited.")
            return 0

        if not user_text:
            continue

        normalized = user_text.lower()
        if normalized in EXIT_COMMANDS:
            print("Exited.")
            return 0

        if normalized in RESET_COMMANDS:
            agent.reset()
            print("Conversation cleared.")
            continue

        try:
            reply = agent.chat(user_text)
        except AgentError as exc:
            print(f"Request failed: {exc}")
            continue

        print(f"Agent> {reply}")


if __name__ == "__main__":
    raise SystemExit(main())
