import re
from collections.abc import Iterator
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from config import Settings
from conversation import ConversationHistory


THINK_BLOCK_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
MAX_ERROR_DETAIL_LENGTH = 300


class AgentError(RuntimeError):
    """User-facing error for chat failures."""


class VoiceTextAgent:
    """A small chat agent backed by an OpenAI-compatible API.

    The agent owns API calls and delegates local conversation storage to
    ConversationHistory. It keeps the two supported API styles separate so the
    command-line interface can switch providers without knowing transport
    details.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the API client and empty conversation state."""
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            default_headers=settings.default_headers or None,
        )
        self._history = ConversationHistory(max_turns=settings.max_history_turns)
        self._previous_response_id: str | None = None

    def reset(self) -> None:
        """Clear local chat history and provider-side Responses API state."""
        self._history.clear()
        self._previous_response_id = None

    def chat(self, user_text: str) -> str:
        """Return the assistant reply for USER_TEXT.

        USER_TEXT must contain non-whitespace text. API-specific exceptions are
        translated into AgentError so callers can show a friendly message
        without importing OpenAI SDK exception classes.
        """
        message = user_text.strip()
        if not message:
            raise ValueError("user_text cannot be empty")

        try:
            if self._settings.api_mode == "responses":
                return self._chat_with_responses_api(message)
            return self._chat_with_chat_completions_api(message)
        except AuthenticationError as exc:
            raise AgentError("API key is invalid or unauthorized.") from exc
        except RateLimitError as exc:
            raise AgentError("Rate limit or quota exceeded. Try again later.") from exc
        except APIConnectionError as exc:
            raise AgentError(
                "Cannot connect to the API service. Check network and Base URL."
            ) from exc
        except APIStatusError as exc:
            detail = _format_status_error(exc)
            raise AgentError(f"API service returned HTTP {exc.status_code}. {detail}") from exc
        except OpenAIError as exc:
            raise AgentError(f"API call failed: {exc}") from exc

    def stream_chat(self, user_text: str) -> Iterator[str]:
        """Yield assistant text chunks for USER_TEXT as the model streams.

        The full assistant reply is committed to history only after streaming
        finishes successfully. Streaming is currently supported for Chat
        Completions because the Responses path relies on provider-managed state.
        """
        message = user_text.strip()
        if not message:
            raise ValueError("user_text cannot be empty")
        if self._settings.api_mode != "chat":
            yield self.chat(message)
            return

        try:
            yield from self._stream_chat_completions_api(message)
        except AuthenticationError as exc:
            raise AgentError("API key is invalid or unauthorized.") from exc
        except RateLimitError as exc:
            raise AgentError("Rate limit or quota exceeded. Try again later.") from exc
        except APIConnectionError as exc:
            raise AgentError(
                "Cannot connect to the API service. Check network and Base URL."
            ) from exc
        except APIStatusError as exc:
            detail = _format_status_error(exc)
            raise AgentError(f"API service returned HTTP {exc.status_code}. {detail}") from exc
        except OpenAIError as exc:
            raise AgentError(f"API call failed: {exc}") from exc

    def _chat_with_chat_completions_api(self, message: str) -> str:
        """Send MESSAGE through Chat Completions and remember the turn."""
        response = self._client.chat.completions.create(
            model=self._settings.model,
            messages=self._history.to_messages(
                system_prompt=self._settings.system_prompt,
                next_user_message=message,
            ),
        )

        reply = _extract_chat_reply(response)
        self._history.add_turn(message, reply)
        return reply

    def _stream_chat_completions_api(self, message: str) -> Iterator[str]:
        """Stream MESSAGE through Chat Completions and remember the full reply."""
        stream = self._client.chat.completions.create(
            model=self._settings.model,
            messages=self._history.to_messages(
                system_prompt=self._settings.system_prompt,
                next_user_message=message,
            ),
            stream=True,
        )

        parts: list[str] = []
        emitted_length = 0
        for event in stream:
            if not event.choices:
                continue

            delta = event.choices[0].delta.content
            if not delta:
                continue

            parts.append(delta)
            cleaned = clean_model_reply("".join(parts))
            if len(cleaned) <= emitted_length:
                continue

            new_text = cleaned[emitted_length:]
            emitted_length = len(cleaned)
            yield new_text

        reply = clean_model_reply("".join(parts))
        if not reply:
            raise AgentError("Model returned no displayable text.")
        self._history.add_turn(message, reply)

    def _chat_with_responses_api(self, message: str) -> str:
        """Send MESSAGE through Responses API using provider-managed context."""
        request: dict[str, Any] = {
            "model": self._settings.model,
            "instructions": self._settings.system_prompt,
            "input": message,
        }
        if self._previous_response_id:
            request["previous_response_id"] = self._previous_response_id

        response = self._client.responses.create(**request)
        self._previous_response_id = response.id

        reply = clean_model_reply(response.output_text or "")
        if not reply:
            raise AgentError("Model returned no displayable text.")
        return reply


def _extract_chat_reply(response) -> str:
    """Extract and clean the first text reply from a Chat Completions response."""
    if not response.choices:
        raise AgentError("Model returned no candidate replies.")

    content = response.choices[0].message.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    else:
        text = ""

    reply = clean_model_reply(text)
    if not reply:
        raise AgentError("Model returned no displayable text.")

    return reply


def clean_model_reply(text: str) -> str:
    """Remove provider-visible reasoning blocks from TEXT before display."""
    stripped = text.lstrip()
    if stripped.startswith("<think>") and "</think>" not in stripped:
        return ""
    return THINK_BLOCK_PATTERN.sub("", text).strip()


def _format_status_error(exc: APIStatusError) -> str:
    """Return a short provider error detail suitable for terminal output."""
    response_text = getattr(exc.response, "text", "") or ""
    response_text = " ".join(response_text.split())
    if not response_text:
        return "The provider did not return error details."

    if len(response_text) > MAX_ERROR_DETAIL_LENGTH:
        response_text = f"{response_text[:MAX_ERROR_DETAIL_LENGTH]}..."

    return response_text
