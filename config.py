from dataclasses import dataclass
import os

from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_SYSTEM_PROMPT = (
    "你是一个语音对话 agent 的文字版原型。"
    "请用自然、简洁、适合口语朗读的中文回答用户。"
    "当用户的问题不清楚时，先问一个简短的澄清问题。"
)


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    system_prompt: str


def load_settings() -> Settings:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "未找到 OPENAI_API_KEY。请先设置环境变量，或复制 .env.example 为 .env 后填写。"
        )

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    system_prompt = (
        os.getenv("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()
        or DEFAULT_SYSTEM_PROMPT
    )

    return Settings(api_key=api_key, model=model, system_prompt=system_prompt)
