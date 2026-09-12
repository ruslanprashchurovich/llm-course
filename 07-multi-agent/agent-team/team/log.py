"""Журнал команды из урока 7.2: общая память плюс телеметрия.

``shape()`` — «форма» журнала (кто -> кому, какого вида) без содержимого и цен.
Именно её сравнивают тесты императивного оркестратора и графа: поведение должно
совпасть узел в узел.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    """Одна реплика в команде: кто, кому, что — и почём."""
    sender: str
    recipient: str
    kind: str          # spec / code / test_report / trace / decision / json / facts
    content: str
    prompt_tokens: int = 0
    answer_tokens: int = 0
    seconds: float = 0.0


@dataclass
class TeamLog:
    messages: list[Message] = field(default_factory=list)
    verbose: bool = True

    def add(self, msg: Message) -> None:
        self.messages.append(msg)
        if self.verbose:
            cost = (f" [{msg.answer_tokens} ток, {msg.seconds:.0f} c]"
                    if msg.answer_tokens else "")
            print(f"  {msg.sender} -> {msg.recipient} ({msg.kind}){cost}")

    def llm_calls(self) -> int:
        return sum(1 for m in self.messages if m.answer_tokens)

    def total_tokens(self) -> tuple[int, int]:
        return (sum(m.prompt_tokens for m in self.messages),
                sum(m.answer_tokens for m in self.messages))

    def shape(self) -> list[tuple[str, str, str]]:
        return [(m.sender, m.recipient, m.kind) for m in self.messages]

    def last(self, kind: str) -> Message | None:
        for msg in reversed(self.messages):
            if msg.kind == kind:
                return msg
        return None
