import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sounddevice as sd

from config import load_settings
from tts import TTSError, create_text_to_speech


DEFAULT_TEXT = (
    "\u4f60\u597d\uff0c"
    "\u8fd9\u662f\u6811\u8393\u6d3e\u8bed\u97f3\u5408\u6210"
    "\u64ad\u653e\u6d4b\u8bd5\u3002"
)


def main() -> int:
    """Synthesize one phrase with TTS and optionally play it locally."""
    parser = argparse.ArgumentParser(description="Test TTS synthesis and playback.")
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to synthesize and play.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Receive TTS audio without opening the local playback device.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print TTS websocket events and audio payload sizes.",
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

    if args.no_play:
        print("Receiving TTS audio without local playback...")
    else:
        print("Playing TTS test audio...")

    try:
        create_text_to_speech(settings.tts).speak_stream(
            [args.text],
            on_text=lambda chunk: print(chunk, end="", flush=True),
            on_event=(lambda event: print(f"\n{event}", flush=True))
            if args.verbose
            else None,
            play_audio=not args.no_play,
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
