"""The sim clock: changing speed mid-run keeps sim time continuous.

    .venv-thor/bin/python -m unittest tests.test_clock
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.clock import SimClock  # noqa: E402


class Speed(unittest.TestCase):
    def test_slowing_down_keeps_time_continuous(self) -> None:
        wall = [100.0]
        with mock.patch("sim.clock.time.monotonic", lambda: wall[0]):
            c = SimClock(1.0)
            wall[0] = 110.0
            self.assertAlmostEqual(c.now(), 10.0)
            c.set_speed(0.5)
            self.assertAlmostEqual(c.now(), 10.0)          # no jump when the speed changes
            wall[0] = 114.0
            self.assertAlmostEqual(c.now(), 12.0)          # 4 wall seconds are 2 sim seconds now
            c.set_speed(2.0)
            wall[0] = 115.0
            self.assertAlmostEqual(c.now(), 14.0)
        with self.assertRaises(ValueError):
            c.set_speed(0)

    def test_pause_stops_time_and_holds_sleeps(self) -> None:
        import asyncio

        async def run():
            c = SimClock(20.0)
            done = []

            async def nap():
                await c.sleep(4.0)                       # 0.2 s of wall time at speed 20
                done.append(c.now())
            task = asyncio.create_task(nap())
            await asyncio.sleep(0.05)
            c.pause()
            t = c.now()
            await asyncio.sleep(0.4)                     # long past when it would have woken
            self.assertEqual(c.now(), t)                 # time stands still
            self.assertEqual(done, [])                   # and the sleep holds
            c.resume()
            await asyncio.wait_for(task, 2)
            self.assertGreaterEqual(done[0], 4.0)
            self.assertFalse(c.paused)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
