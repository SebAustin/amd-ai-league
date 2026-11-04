"""Wall-clock budget shared by every stage of a request."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Deadline:
    """A countdown that never goes negative. Immutable: use ``child`` to carve out sub-budgets."""

    seconds: float
    start: float = field(default_factory=time.monotonic)

    def remaining(self) -> float:
        return max(0.0, self.seconds - (time.monotonic() - self.start))

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def child(self, seconds: float) -> Deadline:
        """A new deadline of ``seconds``, never outliving this one."""
        return Deadline(max(0.0, min(seconds, self.remaining())))
