from dataclasses import dataclass, field


Message = dict[str, str]


@dataclass
class ConversationHistory:
    max_turns: int
    _messages: list[Message] = field(default_factory=list)

    def clear(self) -> None:
        self._messages.clear()

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self._messages.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        self._trim()

    def to_messages(
        self, system_prompt: str, next_user_message: str | None = None
    ) -> list[Message]:
        messages = [{"role": "system", "content": system_prompt}, *self._messages]
        if next_user_message is not None:
            messages.append({"role": "user", "content": next_user_message})
        return messages

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]
