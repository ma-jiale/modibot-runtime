from dataclasses import dataclass
import os

from dotenv import load_dotenv


DEFAULT_PROVIDER = "minimax"
DEFAULT_MODEL = "MiniMax-M2.7"
DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_MAX_HISTORY_TURNS = 20
DEFAULT_API_MODE = "chat"
DEFAULT_TTS_PROVIDER = "system"
DEFAULT_TTS_FORMAT = "wav"
DEFAULT_TTS_OUTPUT_DIR = "outputs"
DEFAULT_TTS_STREAM_CHUNK_CHARS = 60
DEFAULT_ASR_PROVIDER = "faster-whisper"
DEFAULT_ASR_MODEL_SIZE = "base"
DEFAULT_ASR_DEVICE = "cpu"
DEFAULT_ASR_COMPUTE_TYPE = "int8"
DEFAULT_ASR_LANGUAGE = "zh"
DEFAULT_RECORDINGS_DIR = "recordings"
DEFAULT_RECORD_SAMPLE_RATE = 16000
DEFAULT_RECORD_CHANNELS = 1
DEFAULT_VAD_FRAME_MS = 30
DEFAULT_VAD_START_THRESHOLD = 0.018
DEFAULT_VAD_END_THRESHOLD = 0.012
DEFAULT_VAD_SILENCE_MS = 900
DEFAULT_VAD_MIN_SPEECH_MS = 300
DEFAULT_VAD_MAX_RECORD_SECONDS = 20.0
DEFAULT_VAD_PREROLL_MS = 300
DEFAULT_SYSTEM_PROMPT = (
    "You are the text prototype of a voice conversation agent. "
    "Reply in natural, concise Simplified Chinese that is suitable for being read aloud. "
    "Do not use Traditional Chinese unless the user explicitly asks for it. "
    "If the user's request is unclear, ask one short clarifying question first."
)
VALID_API_MODES = {"chat", "responses"}
VALID_TTS_FORMATS = {"mp3", "wav", "pcm", "flac"}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    provider: str
    api_key: str
    model: str
    system_prompt: str
    base_url: str | None
    api_mode: str
    default_headers: dict[str, str]
    max_history_turns: int
    tts: "TTSSettings"
    asr: "ASRSettings"


@dataclass(frozen=True)
class TTSSettings:
    """Runtime configuration for text-to-speech providers."""

    enabled: bool
    provider: str
    file_format: str
    output_dir: str
    autoplay: bool
    streaming: bool
    stream_chunk_chars: int
    speed: float
    volume: float


@dataclass(frozen=True)
class ASRSettings:
    """Runtime configuration for speech-to-text and recording."""

    provider: str
    model_size: str
    device: str
    compute_type: str
    language: str | None
    recordings_dir: str
    sample_rate: int
    channels: int
    vad: "VADSettings"


@dataclass(frozen=True)
class VADSettings:
    """Runtime configuration for voice activity detection."""

    frame_ms: int
    start_threshold: float
    end_threshold: float
    silence_ms: int
    min_speech_ms: int
    max_record_seconds: float
    preroll_ms: int


def load_settings() -> Settings:
    """Load and validate app settings from .env and the process environment.

    MiniMax-specific variables are preferred, while OPENAI_* names remain
    supported so the same code can target other OpenAI-compatible providers.
    """
    load_dotenv()

    provider = os.getenv("AI_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    api_key = _read_first_env("MINIMAX_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "API key is missing. Set MINIMAX_API_KEY or copy .env.example to .env."
        )

    model = _read_first_env("MINIMAX_MODEL", "OPENAI_MODEL") or DEFAULT_MODEL
    system_prompt = (
        os.getenv("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()
        or DEFAULT_SYSTEM_PROMPT
    )

    return Settings(
        provider=provider,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        base_url=_read_base_url(),
        api_mode=_read_api_mode(),
        default_headers=_read_default_headers(),
        max_history_turns=_read_positive_int(
            "MAX_HISTORY_TURNS", DEFAULT_MAX_HISTORY_TURNS
        ),
        tts=_read_tts_settings(),
        asr=_read_asr_settings(),
    )


def _read_tts_settings() -> TTSSettings:
    """Load local text-to-speech settings."""
    provider = _read_first_env("TTS_PROVIDER", default=DEFAULT_TTS_PROVIDER).lower()
    file_format = (
        _read_first_env("TTS_FORMAT", default=DEFAULT_TTS_FORMAT)
        .strip()
        .lower()
    )
    if file_format not in VALID_TTS_FORMATS:
        valid_formats = ", ".join(sorted(VALID_TTS_FORMATS))
        raise RuntimeError(f"TTS_FORMAT must be one of: {valid_formats}.")
    if provider == "system":
        file_format = "wav"

    return TTSSettings(
        enabled=_read_bool("TTS_ENABLED", default=False),
        provider=provider,
        file_format=file_format,
        output_dir=_read_first_env("TTS_OUTPUT_DIR", default=DEFAULT_TTS_OUTPUT_DIR),
        autoplay=_read_bool("TTS_AUTOPLAY", default=False),
        streaming=_read_bool("TTS_STREAMING", default=True),
        stream_chunk_chars=_read_positive_int(
            "TTS_STREAM_CHUNK_CHARS", DEFAULT_TTS_STREAM_CHUNK_CHARS
        ),
        speed=_read_float("TTS_SPEED", default=1.0),
        volume=_read_float("TTS_VOLUME", default=1.0),
    )


def _read_asr_settings() -> ASRSettings:
    """Load speech-to-text and recorder settings."""
    language = _read_first_env("ASR_LANGUAGE", default=DEFAULT_ASR_LANGUAGE)
    return ASRSettings(
        provider=_read_first_env("ASR_PROVIDER", default=DEFAULT_ASR_PROVIDER).lower(),
        model_size=_read_first_env("ASR_MODEL_SIZE", default=DEFAULT_ASR_MODEL_SIZE),
        device=_normalize_asr_device(
            _read_first_env("ASR_DEVICE", default=DEFAULT_ASR_DEVICE)
        ),
        compute_type=_read_first_env(
            "ASR_COMPUTE_TYPE", default=DEFAULT_ASR_COMPUTE_TYPE
        ),
        language=language or None,
        recordings_dir=_read_first_env(
            "RECORDINGS_DIR", default=DEFAULT_RECORDINGS_DIR
        ),
        sample_rate=_read_positive_int(
            "RECORD_SAMPLE_RATE", DEFAULT_RECORD_SAMPLE_RATE
        ),
        channels=_read_positive_int("RECORD_CHANNELS", DEFAULT_RECORD_CHANNELS),
        vad=_read_vad_settings(),
    )


def _read_vad_settings() -> VADSettings:
    """Load energy-based VAD settings."""
    return VADSettings(
        frame_ms=_read_positive_int("VAD_FRAME_MS", DEFAULT_VAD_FRAME_MS),
        start_threshold=_read_float(
            "VAD_START_THRESHOLD", DEFAULT_VAD_START_THRESHOLD
        ),
        end_threshold=_read_float("VAD_END_THRESHOLD", DEFAULT_VAD_END_THRESHOLD),
        silence_ms=_read_positive_int("VAD_SILENCE_MS", DEFAULT_VAD_SILENCE_MS),
        min_speech_ms=_read_positive_int(
            "VAD_MIN_SPEECH_MS", DEFAULT_VAD_MIN_SPEECH_MS
        ),
        max_record_seconds=_read_float(
            "VAD_MAX_RECORD_SECONDS", DEFAULT_VAD_MAX_RECORD_SECONDS
        ),
        preroll_ms=_read_positive_int("VAD_PREROLL_MS", DEFAULT_VAD_PREROLL_MS),
    )


def _read_base_url() -> str | None:
    """Return the normalized OpenAI-compatible API base URL."""
    raw_url = _read_first_env(
        "MINIMAX_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        default=DEFAULT_BASE_URL,
    )
    return _normalize_base_url(raw_url)


def _normalize_asr_device(device: str) -> str:
    """Return the faster-whisper device name for common user aliases."""
    normalized = device.strip().lower()
    if normalized == "gpu":
        return "cuda"
    return normalized


def _read_api_mode() -> str:
    """Return the selected API mode after validating it."""
    api_mode = (
        _read_first_env("MINIMAX_API_MODE", "OPENAI_API_MODE", default=DEFAULT_API_MODE)
        .strip()
        .lower()
    )
    if api_mode not in VALID_API_MODES:
        valid_modes = ", ".join(sorted(VALID_API_MODES))
        raise RuntimeError(f"MINIMAX_API_MODE must be one of: {valid_modes}.")
    return api_mode


def _read_default_headers() -> dict[str, str]:
    """Return optional provider headers, omitting unset values."""
    headers = {
        "User-Agent": _read_first_env("MINIMAX_USER_AGENT", "OPENAI_USER_AGENT"),
        "HTTP-Referer": _read_first_env(
            "MINIMAX_HTTP_REFERER", "OPENAI_HTTP_REFERER"
        ),
        "X-Title": _read_first_env("MINIMAX_X_TITLE", "OPENAI_X_TITLE"),
    }
    return {key: value for key, value in headers.items() if value}


def _read_positive_int(name: str, default: int) -> int:
    """Return environment variable NAME as a positive int, or DEFAULT."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc

    if value <= 0:
        raise RuntimeError(f"{name} must be greater than 0.")

    return value


def _read_float(name: str, default: float) -> float:
    """Return environment variable NAME as a float, or DEFAULT."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number.") from exc


def _read_bool(name: str, default: bool) -> bool:
    """Return environment variable NAME as a bool, or DEFAULT."""
    raw_value = os.getenv(name, "").strip().lower()
    if not raw_value:
        return default

    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False

    raise RuntimeError(f"{name} must be true or false.")


def _normalize_base_url(raw_url: str) -> str | None:
    """Normalize RAW_URL to the API root expected by the OpenAI SDK.

    Users sometimes paste a full endpoint path. The SDK appends endpoint paths
    itself, so this function trims known suffixes like /chat/completions.
    """
    if not raw_url:
        return None

    base_url = raw_url.rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]

    return base_url


def _read_first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment value among NAMES."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default.strip()
