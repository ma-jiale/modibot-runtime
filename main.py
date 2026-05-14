import asyncio

from agent import AgentError, VoiceTextAgent
from config import load_settings
from streaming_tts import speak_streaming_response
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
        print(f"TTS: {settings.tts.provider}")
        print(f"TTS streaming: {settings.tts.streaming}")
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

        _handle_user_turn(agent, tts_provider, user_text, settings.tts)


def _handle_user_turn(agent: VoiceTextAgent, tts_provider, user_text: str, tts) -> None:
    """Route one user turn through text-only, streaming TTS, or file TTS."""
    if tts_provider is None:
        _reply_with_text(agent, user_text)
    elif tts.streaming:
        _stream_reply_with_tts(agent, tts_provider, user_text, tts.stream_chunk_chars)
    else:
        _reply_with_file_tts(agent, tts_provider, user_text, tts.autoplay)


def _reply_with_text(agent: VoiceTextAgent, user_text: str) -> None:
    """Print one non-streaming assistant reply."""
    try:
        reply = agent.chat(user_text)
    except AgentError as exc:
        print(f"Request failed: {exc}")
        return

    print(f"Agent> {reply}")


def _reply_with_file_tts(
    agent: VoiceTextAgent, tts_provider, user_text: str, autoplay: bool
) -> None:
    """Generate one full reply, save it as audio, and optionally play it."""
    try:
        reply = agent.chat(user_text)
        print(f"Agent> {reply}")
        audio = tts_provider.synthesize_to_file(reply)
    except (AgentError, TTSError) as exc:
        print(f"Request failed: {exc}")
        return

    print(f"TTS saved: {audio.path}")
    if autoplay:
        play_audio_file(audio.path)


def _stream_reply_with_tts(
    agent: VoiceTextAgent, tts_provider, user_text: str, chunk_chars: int
) -> None:
    """Stream text into TTS audio bytes and play them in order."""
    print("Agent> ", end="", flush=True)
    try:
        asyncio.run(
            speak_streaming_response(
                text_stream=agent.stream_chat(user_text),
                provider=tts_provider,
                max_chunk_chars=chunk_chars,
            )
        )
        print()
    except (AgentError, TTSError) as exc:
        print(f"\nRequest failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
