from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from config import Settings


class AgentError(RuntimeError):
    """User-facing error for chat failures."""


class VoiceTextAgent:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.api_key)
        self._previous_response_id: str | None = None

    @property
    def previous_response_id(self) -> str | None:
        return self._previous_response_id

    def reset(self) -> None:
        self._previous_response_id = None

    def chat(self, user_text: str) -> str:
        message = user_text.strip()
        if not message:
            raise ValueError("user_text cannot be empty")

        request = {
            "model": self._settings.model,
            "instructions": self._settings.system_prompt,
            "input": message,
        }
        if self._previous_response_id:
            request["previous_response_id"] = self._previous_response_id

        try:
            response = self._client.responses.create(**request)
        except AuthenticationError as exc:
            raise AgentError("OpenAI API Key 无效或没有权限，请检查 OPENAI_API_KEY。") from exc
        except RateLimitError as exc:
            raise AgentError("触发 OpenAI 速率限制或额度不足，请稍后再试。") from exc
        except APIConnectionError as exc:
            raise AgentError("无法连接 OpenAI API，请检查网络连接。") from exc
        except APIStatusError as exc:
            raise AgentError(f"OpenAI API 返回错误：HTTP {exc.status_code}") from exc
        except OpenAIError as exc:
            raise AgentError(f"OpenAI API 调用失败：{exc}") from exc

        self._previous_response_id = response.id
        text = (response.output_text or "").strip()
        if not text:
            raise AgentError("模型没有返回可显示的文本。")
        return text
