from dataclasses import dataclass
import os

from dotenv import load_dotenv


DEFAULT_PROVIDER = "minimax"
DEFAULT_MODEL = "MiniMax-M2.7"
DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_MAX_HISTORY_TURNS = 20
DEFAULT_API_MODE = "chat"
DEFAULT_SYSTEM_PROMPT = (
    "You are the text prototype of a voice conversation agent. "
    "Reply in natural, concise Chinese that is suitable for being read aloud. "
    "If the user's request is unclear, ask one short clarifying question first."
)
VALID_API_MODES = {"chat", "responses"}


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
