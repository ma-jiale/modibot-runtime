import asyncio
from dataclasses import dataclass

from agent import AgentError, VoiceTextAgent
from asr import ASRError, SpeechRecognizer, create_speech_recognizer
from config import Settings, TTSSettings, load_settings
from recorder import RecorderError, WavRecorder
from streaming_tts import speak_streaming_response
from tts import TTSError, TextToSpeechProvider, create_tts_provider, play_audio_file


EXIT_COMMANDS = {"exit", "quit", "q", "bye", "tuichu", "zaijian"}
RESET_COMMANDS = {"reset", "/reset", "clear", "qingkong", "chongzhi"}
VOICE_COMMANDS = {"voice", "v", "speak"}


@dataclass(frozen=True)
class Runtime:
    """The long-lived services used by the command-line loop."""

    settings: Settings
    agent: VoiceTextAgent
    tts_provider: TextToSpeechProvider | None
    recorder: WavRecorder
    recognizer: SpeechRecognizer


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

    runtime = _create_runtime(settings)
    _print_startup(runtime.settings)

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
            runtime.agent.reset()
            print("Conversation cleared.")
            continue

        if normalized in VOICE_COMMANDS:
            user_text = _record_and_transcribe(runtime.recorder, runtime.recognizer)
            if not user_text:
                continue

        _handle_user_turn(runtime, user_text)


def _create_runtime(settings: Settings) -> Runtime:
    """Create long-lived services from SETTINGS."""
    return Runtime(
        settings=settings,
        agent=VoiceTextAgent(settings),
        tts_provider=(
            create_tts_provider(settings.tts) if settings.tts.enabled else None
        ),
        recorder=WavRecorder(settings.asr),
        recognizer=create_speech_recognizer(settings.asr),
    )


def _print_startup(settings: Settings) -> None:
    """Print configuration that is safe to show in the terminal."""
    print("Text conversation agent started.")
    print(f"Provider: {settings.provider}")
    print(f"Model: {settings.model}")
    print(f"API mode: {settings.api_mode}")
    if settings.base_url:
        print(f"Base URL: {settings.base_url}")
    if settings.tts.enabled:
        print(f"TTS: {settings.tts.provider}")
        print(f"TTS streaming: {settings.tts.streaming}")
    print("Type voice to record, exit/quit to stop, or reset/clear to clear history.")


def _record_and_transcribe(
    recorder: WavRecorder, recognizer: SpeechRecognizer
) -> str | None:
    """Record one utterance, transcribe it, and return text."""
    try:
        recording = recorder.record_until_enter()
        print(f"Recorded: {recording.path}")
        result = recognizer.transcribe(recording.path)
    except (RecorderError, ASRError) as exc:
        print(f"Voice input failed: {exc}")
        return None

    print(f"You said> {result.text}")
    return result.text


def _handle_user_turn(runtime: Runtime, user_text: str) -> None:
    """Route one user turn through text-only, streaming TTS, or file TTS."""
    if runtime.tts_provider is None:
        _reply_with_text(runtime.agent, user_text)
    elif runtime.settings.tts.streaming:
        _stream_reply_with_tts(
            runtime.agent,
            runtime.tts_provider,
            user_text,
            runtime.settings.tts.stream_chunk_chars,
        )
    else:
        _reply_with_file_tts(
            runtime.agent,
            runtime.tts_provider,
            user_text,
            runtime.settings.tts.autoplay,
        )


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
