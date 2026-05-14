from dataclasses import dataclass, field


Message = dict[str, str]


@dataclass
class ConversationHistory:
    """A bounded list of user/assistant messages.

    max_turns counts complete user+assistant exchanges. The class stores only
    local Chat Completions history; Responses API state is tracked separately by
    the agent through previous_response_id.
    """

    max_turns: int
    _messages: list[Message] = field(default_factory=list)

    def clear(self) -> None:
        """Remove all remembered messages."""
        self._messages.clear()

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """Append one user/assistant exchange and enforce the history limit."""
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
        """Return SDK-ready messages with SYSTEM_PROMPT at the front.

        If NEXT_USER_MESSAGE is given, it is appended without mutating history.
        This lets the agent send a trial request before committing the turn.
        """
        messages = [{"role": "system", "content": system_prompt}, *self._messages]
        if next_user_message is not None:
            messages.append({"role": "user", "content": next_user_message})
        return messages

    def _trim(self) -> None:
        """Keep only the newest max_turns exchanges."""
        max_messages = self.max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]
