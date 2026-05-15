from dataclasses import dataclass

from agent import AgentError, VoiceTextAgent
from asr import ASRError, SpeechRecognizer, create_speech_recognizer
from config import Settings, load_settings
from recorder import RecorderError, WavRecorder
from tts import TTSError, TextToSpeech, create_text_to_speech
from voice_activity import VADError, VoiceActivityDetector, create_vad


EXIT_COMMANDS = {"exit", "quit", "q", "bye", "tuichu", "zaijian"}
RESET_COMMANDS = {"reset", "/reset", "clear", "qingkong", "chongzhi"}
VOICE_COMMANDS = {"voice", "v", "speak"}
VOICE_EXIT_PHRASES = {
    "\u9000\u51fa",
    "\u9000\u51fa\u8bed\u97f3",
    "\u9000\u51fa\u8bed\u97f3\u6a21\u5f0f",
    "\u505c\u6b62",
    "\u7ed3\u675f",
    "\u518d\u89c1",
}


@dataclass(frozen=True)
class Runtime:
    """The long-lived services used by the command-line loop."""

    settings: Settings
    agent: VoiceTextAgent
    recorder: WavRecorder
    recognizer: SpeechRecognizer
    vad: VoiceActivityDetector
    tts: TextToSpeech


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

    try:
        runtime = _create_runtime(settings)
    except VADError as exc:
        print(f"VAD error: {exc}")
        return 1
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
            _run_voice_loop(runtime)
            continue

        _handle_user_turn(runtime, user_text)


def _run_voice_loop(runtime: Runtime) -> None:
    """Continuously listen for utterances until a spoken exit phrase."""
    print("Voice mode started. Say the voice-exit phrase to stop.")
    while True:
        user_text = _record_and_transcribe(
            runtime.recorder, runtime.recognizer, runtime.vad
        )
        if not user_text:
            continue

        if _is_voice_exit(user_text):
            print("Voice mode stopped.")
            return

        _handle_user_turn(runtime, user_text)


def _is_voice_exit(text: str) -> bool:
    """Return whether TEXT asks to leave voice mode."""
    normalized = (
        text.strip()
        .replace("\u3002", "")
        .replace("\uff01", "")
        .replace("\uff1f", "")
    )
    return normalized in VOICE_EXIT_PHRASES


def _create_runtime(settings: Settings) -> Runtime:
    """Create long-lived services from SETTINGS."""
    return Runtime(
        settings=settings,
        agent=VoiceTextAgent(settings),
        recorder=WavRecorder(settings.asr),
        recognizer=create_speech_recognizer(settings.asr),
        vad=create_vad(settings.asr.vad),
        tts=create_text_to_speech(settings.tts),
    )


def _print_startup(settings: Settings) -> None:
    """Print configuration that is safe to show in the terminal."""
    print("Raspberry Pi voice agent started.")
    print(f"Provider: {settings.provider}")
    print(f"Model: {settings.model}")
    print(f"API mode: {settings.api_mode}")
    if settings.base_url:
        print(f"Base URL: {settings.base_url}")
    print(f"ASR: {settings.asr.provider}")
    if settings.asr.remote_url:
        print(f"ASR URL: {settings.asr.remote_url}")
    if settings.asr.record_device:
        print(f"Microphone: {settings.asr.record_device}")
    print("VAD: TEN VAD")
    print(f"TTS: {settings.tts.provider}")
    if settings.tts.provider != "none":
        print(f"TTS speaker: {settings.tts.speaker}")
        print(f"TTS output: {settings.tts.output_device or 'system default'}")
    print("Type voice to record, exit/quit to stop, or reset/clear to clear history.")


def _record_and_transcribe(
    recorder: WavRecorder,
    recognizer: SpeechRecognizer,
    vad: VoiceActivityDetector,
) -> str | None:
    """Record one utterance, transcribe it, and return text."""
    try:
        recording = recorder.record_until_silence(vad)
        print(f"Recorded: {recording.path}")
        result = recognizer.transcribe(recording.path)
    except (RecorderError, ASRError) as exc:
        print(f"Voice input failed: {exc}")
        return None

    print(f"You said> {result.text}")
    return result.text


def _handle_user_turn(runtime: Runtime, user_text: str) -> None:
    """Route one user turn through the assistant and TTS playback."""
    _reply_with_text(runtime.agent, runtime.tts, user_text)


def _reply_with_text(agent: VoiceTextAgent, tts: TextToSpeech, user_text: str) -> None:
    """Stream one assistant reply to the terminal and speech output."""
    try:
        print("Agent> ", end="", flush=True)
        tts.speak_stream(
            agent.stream_chat(user_text),
            on_text=lambda chunk: print(chunk, end="", flush=True),
        )
    except AgentError as exc:
        print()
        print(f"Request failed: {exc}")
        return
    except TTSError as exc:
        print()
        print(f"TTS failed: {exc}")
        return

    print()


if __name__ == "__main__":
    raise SystemExit(main())
