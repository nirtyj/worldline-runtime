"""The robot learns what it sees from its perception source, never from the world.

A source that sees nothing makes a look come back empty and a reach unknown, though the
synthetic room (tests/thor_synthetic.py) has the things in plain view.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception.source import StandIn  # noqa: E402
from sim.clock import SimClock  # noqa: E402
from sim.goals import GoalRejected  # noqa: E402
from sim.log import EventLog  # noqa: E402
from sim.robot import SimRobot  # noqa: E402
from tests.thor_synthetic import SyntheticController  # noqa: E402
from thor import ThorWorld  # noqa: E402


class Blind:
    """Sees nothing, knows one made-up id."""
    source = "blind"

    def objects(self):
        return {}

    def landmarks(self):
        return {}

    async def view(self):
        return {}, {}

    def knows(self, oid):
        return oid == "ghost_1"

    def target(self, oid):
        return None


class PerceptionSeam(unittest.TestCase):
    def setUp(self) -> None:
        self.world = ThorWorld(width=320, height=240)
        self.addCleanup(self.world.close)
        with patch("ai2thor.controller.Controller", SyntheticController):
            self.world.load("FloorPlan1")

    def robot(self, perception=None) -> SimRobot:
        clock = SimClock(speed=50.0)        # the skills' timed steps pass quickly
        return SimRobot(self.world, clock, EventLog(clock), perception)

    def test_the_default_is_the_stand_in(self) -> None:
        robot = self.robot()
        self.assertIsInstance(robot.perceiver, StandIn)
        self.assertEqual(set(robot.perception()["objects"]), {"apple_1", "bowl_1", "mug_1", "mug_2"})
        self.assertEqual(robot.perception()["source"], "thor-ground-truth")

    def test_a_blind_robot_reports_nothing(self) -> None:
        p = self.robot(Blind()).perception()
        self.assertEqual((p["objects"], p["landmarks"], p["source"]), ({}, {}, "blind"))

    def test_look_and_reach_go_through_the_source(self) -> None:
        async def run():
            robot = self.robot(Blind())
            look = await robot.send_goal("look").result()
            reach = await robot.send_goal("reachability", object="ghost_1").result()
            with self.assertRaises(GoalRejected):
                robot.send_goal("reachability", object="apple_1")     # in the room, but not an id it knows
            return look, reach
        look, reach = asyncio.run(run())
        self.assertEqual((look.status, look.data["surfaces"], look.data["landmarks"]), ("SUCCEEDED", {}, []))
        self.assertEqual((reach.data["reachable"], reach.data["reason"]), (False, "not_found"))

    def test_stand_in_look_sees_the_room(self) -> None:
        async def run():
            return await self.robot().send_goal("look").result()
        look = asyncio.run(run())
        seen = {o["id"] for items in look.data["surfaces"].values() for o in items}
        self.assertEqual(seen, {"apple_1", "bowl_1", "mug_1"})          # the mug in the hand isn't on a surface
        self.assertEqual([lm["id"] for lm in look.data["landmarks"]], ["coffee_machine_1", "fridge_1"])


if __name__ == "__main__":
    unittest.main()
