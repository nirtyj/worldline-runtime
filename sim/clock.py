"""One clock for the whole session.

Time is measured in seconds from the start of the session. Everything that waits
-- the robot, the runtime, the server -- waits through this clock, never through
``time.sleep`` or a bare ``asyncio.sleep``, so timeouts agree. The page's Speed
setting slows it down to talk over it; the models still answer in wall time, and Pause
stops sim time altogether.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, TypeVar

T = TypeVar("T")


class SimClock:
    def __init__(self, speed: float = 1.0) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = float(speed)
        self._t0 = time.monotonic()
        self._frozen: float | None = None          # sim time while paused
        self._running = asyncio.Event()
        self._running.set()

    def now(self) -> float:
        """Sim seconds since the scenario started (it stands still while paused)."""
        if self._frozen is not None:
            return self._frozen
        return (time.monotonic() - self._t0) * self.speed

    @property
    def paused(self) -> bool:
        return self._frozen is not None

    def pause(self) -> None:
        """Stop sim time: every wait through this clock holds until resume(), so nothing
        moves, speaks or decides (model calls already on their way still come back)."""
        if self._frozen is None:
            self._frozen = self.now()
            self._running.clear()

    def resume(self) -> None:
        if self._frozen is not None:
            self._t0 = time.monotonic() - self._frozen / self.speed
            self._frozen = None
            self._running.set()

    def set_speed(self, speed: float) -> None:
        """Run faster or slower from now on. Sim time carries on from where it is."""
        if speed <= 0:
            raise ValueError("speed must be positive")
        now = self.now()
        self.speed = float(speed)
        if self._frozen is None:
            self._t0 = time.monotonic() - now / self.speed

    async def sleep(self, seconds: float) -> None:
        """Sleep for ``seconds`` of sim time (yields even for 0). While paused, sim time
        stands still, so the sleep waits for resume()."""
        if self._frozen is not None:
            await self._running.wait()
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        end = self.now() + seconds
        while True:
            if self._frozen is not None:
                await self._running.wait()
            left = end - self.now()
            if left <= 0:
                return
            await asyncio.sleep(min(left / self.speed, 0.25))     # short enough to notice a pause

    async def sleep_until(self, t: float) -> None:
        await self.sleep(t - self.now())

    async def wait_for(self, aw: Awaitable[T], timeout: float) -> T:
        """``asyncio.wait_for`` with a timeout in sim seconds."""
        return await asyncio.wait_for(aw, timeout / self.speed)

    def timeout(self, seconds: float):
        """``asyncio.timeout`` with a delay in sim seconds (Python 3.11+)."""
        return asyncio.timeout(seconds / self.speed)
