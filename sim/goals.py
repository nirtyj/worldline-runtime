"""Goals: how the runtime starts, cancels and awaits a robot skill (ROS 2 action style).

    goal = robot.send_goal("navigate", to="counter_2a")   # may raise GoalRejected
    goal.cancel()                  # a REQUEST: the skill stops at its next safe point
    result = await goal.result()   # GoalResult(status, data); status is
                                   # SUCCEEDED, ABORTED or CANCELED

A robot that hands these out must provide ``_goal_ids`` (an id counter),
``clock`` and ``_public(type, **fields)`` (to announce the cancel request).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


class GoalRejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class GoalResult:
    status: str                     # SUCCEEDED / ABORTED / CANCELED
    data: dict[str, Any] = field(default_factory=dict)


class Goal:
    def __init__(self, robot: Any, skill: str, args: dict[str, Any]) -> None:
        self.id = next(robot._goal_ids)
        self.skill = skill
        self.args = dict(args)
        self.status = "EXECUTING"
        self.created_t = robot.clock.now()
        self._robot = robot
        self._cancel = asyncio.Event()
        self._future: asyncio.Future = asyncio.get_running_loop().create_future()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def done(self) -> bool:
        return self._future.done()

    def cancel(self) -> None:
        """Request cancellation. Returns immediately; await ``result()``."""
        if self.done or self._cancel.is_set():
            return
        self._cancel.set()
        self.status = "CANCELING"
        self._robot._public("cancel_requested", goal=self.id, skill=self.skill)

    async def result(self) -> GoalResult:
        return await asyncio.shield(self._future)

    def __repr__(self) -> str:
        return f"<Goal #{self.id} {self.skill}({self.args}) {self.status}>"
