from agent import AgentError, VoiceTextAgent
from config import load_settings
from tts import TTSError, create_tts_provider, play_audio_file


EXIT_COMMANDS = {"exit", "quit", "q", "bye", "tuichu", "zaijian"}
RESET_COMMANDS = {"reset", "/reset", "clear", "qingkong", "chongzhi"}


def main() -> int:
    """Run the command-line chat loop.

    The CLI is intentionally thin: configuration, API calls, and history are
    handled by helper modules so this function only coordinates user input and
    terminal output.
    """
    try:
        settings = load_settings()
    except RuntimeError as exc:
        print(f"Config error: {exc}")
        return 1

    agent = VoiceTextAgent(settings)
    tts_provider = create_tts_provider(settings.tts) if settings.tts.enabled else None

    print("Text conversation agent started.")
    print(f"Provider: {settings.provider}")
    print(f"Model: {settings.model}")
    print(f"API mode: {settings.api_mode}")
    if settings.base_url:
        print(f"Base URL: {settings.base_url}")
    if settings.tts.enabled:
        print(f"TTS: {settings.tts.provider} / {settings.tts.model}")
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
        if tts_provider is None:
            continue

        try:
            audio = tts_provider.synthesize_to_file(reply)
        except TTSError as exc:
            print(f"TTS failed: {exc}")
            continue

        print(f"TTS saved: {audio.path}")
        if settings.tts.autoplay:
            play_audio_file(audio.path)


if __name__ == "__main__":
    raise SystemExit(main())
