"""The layout and the robot's read side, pinned against a made-up THOR room.

tests/fixtures/layout_*.json were written by the code as it was before the layout
logic moved from thor/world.py to sim/layout.py; they must not change. To rewrite
them on purpose: ``python tests/test_layout_characterization.py --write``.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.clock import SimClock  # noqa: E402
from sim.log import EventLog  # noqa: E402
from tests.thor_synthetic import HOUSE, HOUSE_NAME, SyntheticController, snapshot  # noqa: E402
from thor import ThorRobot, ThorWorld  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CASES = {"room": "FloorPlan1", "house": HOUSE_NAME}


def load(scene: str) -> dict:
    world = ThorWorld(width=320, height=240)
    try:
        with patch("ai2thor.controller.Controller", SyntheticController), \
                patch("thor.procthor.load_house", lambda name: HOUSE):
            world.load(scene)
        clock = SimClock()
        return snapshot(world, ThorRobot(world, clock, EventLog(clock)))
    finally:
        world.close()


class LayoutCharacterization(unittest.TestCase):
    maxDiff = None

    def check(self, case: str) -> None:
        want = json.loads((FIXTURES / f"layout_{case}.json").read_text())
        got = load(CASES[case])
        for key in want:
            self.assertEqual(got[key], want[key], key)
        self.assertEqual(sorted(got), sorted(want))

    def test_room(self) -> None:
        self.check("room")

    def test_house(self) -> None:
        self.check("house")


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURES.mkdir(exist_ok=True)
        for case, scene in CASES.items():
            (FIXTURES / f"layout_{case}.json").write_text(json.dumps(load(scene), indent=1, sort_keys=True) + "\n")
            print("wrote", FIXTURES / f"layout_{case}.json")
    else:
        unittest.main()
