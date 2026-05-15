import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sounddevice as sd

from config import load_settings
from tts import TTSError, create_text_to_speech


def main() -> int:
    """Synthesize one phrase with TTS and play it through sounddevice."""
    parser = argparse.ArgumentParser(description="Test TTS synthesis and playback.")
    parser.add_argument(
        "--text",
        default="你好，这是树莓派语音合成播放测试。",
        help="Text to synthesize and play.",
    )
    args = parser.parse_args()

    try:
        settings = load_settings()
    except RuntimeError as exc:
        print(f"Config error: {exc}")
        return 1

    print("Audio devices visible to sounddevice:")
    print(sd.query_devices())
    print()
    print(f"TTS provider: {settings.tts.provider}")
    print(f"TTS speaker: {settings.tts.speaker}")
    print(f"TTS output: {settings.tts.output_device or 'system default'}")
    print(f"TTS audio: {settings.tts.audio_format}, {settings.tts.sample_rate} Hz")
    print()

    if settings.tts.provider == "none":
        print("TTS is disabled. Set TTS_PROVIDER=doubao before running this check.")
        return 1

    print("Playing TTS test audio...")
    try:
        create_text_to_speech(settings.tts).speak_stream(
            [args.text],
            on_text=lambda chunk: print(chunk, end="", flush=True),
        )
    except TTSError as exc:
        print()
        print(f"TTS failed: {exc}")
        return 1

    print()
    print("TTS playback test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
