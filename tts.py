import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from config import TTSSettings


class TTSError(RuntimeError):
    """User-facing error for text-to-speech failures."""


@dataclass(frozen=True)
class SynthesisResult:
    """The bytes and saved path produced by a TTS request."""

    audio_bytes: bytes
    path: Path
    format: str


class TextToSpeechProvider(Protocol):
    """Common interface implemented by concrete TTS providers."""

    def synthesize_to_bytes(self, text: str) -> bytes:
        """Convert TEXT to speech and return WAV audio bytes."""

    def synthesize_to_file(self, text: str) -> SynthesisResult:
        """Convert TEXT to speech and save it to a local audio file."""


class SystemSpeechTTSProvider:
    """Windows system TTS provider using PowerShell System.Speech.

    This provider needs no cloud TTS plan. It writes a WAV file with the voices
    installed on the user's Windows machine.
    """

    def __init__(self, settings: TTSSettings) -> None:
        self._settings = settings

    def synthesize_to_file(self, text: str) -> SynthesisResult:
        """Synthesize TEXT with the local Windows speech engine."""
        content = text.strip()
        if not content:
            raise ValueError("text cannot be empty")

        if sys.platform != "win32":
            raise TTSError("The system TTS provider currently supports Windows only.")

        audio_bytes = self.synthesize_to_bytes(content)
        output_path = self._write_audio_file(audio_bytes)
        return SynthesisResult(audio_bytes=audio_bytes, path=output_path, format="wav")

    def synthesize_to_bytes(self, text: str) -> bytes:
        """Synthesize TEXT and return WAV bytes without keeping a file."""
        content = text.strip()
        if not content:
            raise ValueError("text cannot be empty")

        if sys.platform != "win32":
            raise TTSError("The system TTS provider currently supports Windows only.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            self._run_powershell_synthesis(content, temp_path)
            return temp_path.read_bytes()
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _build_output_path(self) -> Path:
        """Return a unique WAV output path."""
        output_dir = Path(self._settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return output_dir / f"tts_{timestamp}.wav"

    def _write_audio_file(self, audio_bytes: bytes) -> Path:
        """Save AUDIO_BYTES to the configured output directory."""
        output_path = self._build_output_path()
        output_path.write_bytes(audio_bytes)
        return output_path

    def _run_powershell_synthesis(self, text: str, output_path: Path) -> None:
        """Invoke PowerShell with environment values instead of interpolating TEXT."""
        script = (
            "$Text = $env:VOICE_AGENT_TTS_TEXT;"
            "$Path = $env:VOICE_AGENT_TTS_PATH;"
            "$Rate = [int]$env:VOICE_AGENT_TTS_RATE;"
            "$Volume = [int]$env:VOICE_AGENT_TTS_VOLUME;"
            "Add-Type -AssemblyName System.Speech;"
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$synth.Rate = [Math]::Max(-10, [Math]::Min(10, $Rate));"
            "$synth.Volume = [Math]::Max(0, [Math]::Min(100, $Volume));"
            "$synth.SetOutputToWaveFile($Path);"
            "$synth.Speak($Text);"
            "$synth.Dispose();"
        )
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ]
        env = os.environ.copy()
        env.update(
            {
                "VOICE_AGENT_TTS_TEXT": text,
                "VOICE_AGENT_TTS_PATH": str(output_path.resolve()),
                "VOICE_AGENT_TTS_RATE": str(_system_rate_from_speed(self._settings.speed)),
                "VOICE_AGENT_TTS_VOLUME": str(
                    _system_volume_from_volume(self._settings.volume)
                ),
            }
        )

        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            env=env,
        )
        if completed.returncode != 0:
            detail = " ".join((completed.stderr or "").split())
            raise TTSError(f"System TTS failed: {detail}")


def create_tts_provider(settings: TTSSettings) -> TextToSpeechProvider:
    """Create a concrete TTS provider from SETTINGS."""
    if settings.provider == "system":
        return SystemSpeechTTSProvider(settings)
    raise TTSError(f"Unsupported TTS provider: {settings.provider}")


def play_audio_file(path: Path) -> None:
    """Open PATH with the OS default audio player."""
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
        return

    command = (
        ["open", str(path)]
        if os.uname().sysname == "Darwin"
        else ["xdg-open", str(path)]
    )
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def play_audio_bytes_blocking(audio_bytes: bytes) -> None:
    """Play WAV AUDIO_BYTES and wait until playback finishes."""
    if os.name != "nt":
        raise TTSError("In-memory audio playback currently supports Windows only.")

    try:
        import winsound

        winsound.PlaySound(audio_bytes, winsound.SND_MEMORY)
    except RuntimeError as exc:
        raise TTSError(f"Audio playback failed: {exc}") from exc


def _system_rate_from_speed(speed: float) -> int:
    """Map a provider-neutral speed value to Windows SpeechSynthesizer rate."""
    return round((speed - 1.0) * 5)


def _system_volume_from_volume(volume: float) -> int:
    """Map a provider-neutral volume value to Windows SpeechSynthesizer volume."""
    return round(volume * 100)
