"""The robot's soul: personas/robot.md, read by its own LLM; quiet starts nothing.

    .venv-thor/bin/python -m unittest tests.test_soul
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.soul import SoulPersona, load_soul, parse_soul, write_soul  # noqa: E402
from agent.state import BeliefState  # noqa: E402


class FakeClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def tool_call(self, system: str, text: str, tools: list) -> object:
        self.prompts.append(text)
        return SimpleNamespace(name="intend", args={"text": "look around the kitchen"}, input_tokens=1, output_tokens=1)


class RobotSoul(unittest.TestCase):
    def test_the_robot_has_a_soul(self) -> None:
        soul = load_soul("robot")
        self.assertIn("navigate", soul.autonomy)
        self.assertTrue(soul.prose.startswith("#"))

    def test_quiet_starts_nothing_and_not_quiet_picks_a_goal(self) -> None:
        async def run():
            client = FakeClient()
            p = SoulPersona(load_soul("robot"), client, name="robot", level="off")
            self.assertIsNone(p.propose(BeliefState(), {"keypoints": {}}, 20.0, quiet_s=30.0))
            self.assertEqual(client.prompts, [])                     # quiet: never asks its LLM
            p.level = "medium"                                       # the Quiet switch off
            p.propose(BeliefState(), {"keypoints": {}}, 21.0, quiet_s=30.0)
            await asyncio.sleep(0)
            await p._task
            return p.propose(BeliefState(), {"keypoints": {}}, 22.0, quiet_s=30.0)
        g = asyncio.run(run())
        self.assertEqual(g.drive, "soul")
        self.assertIn("look around the kitchen", g.text)


class WriterClient:
    """Answers the soul writer the way a model might, untidily."""

    def __init__(self, args: dict) -> None:
        self.args = args

    async def tool_call(self, system: str, text: str, tools: list) -> object:
        return SimpleNamespace(name="write_soul", args=self.args, input_tokens=1, output_tokens=1)


class CustomSoul(unittest.TestCase):
    def test_every_menu_soul_loads(self) -> None:
        for name in ("robot", "robot_chatty", "robot_cleaning", "robot_security"):
            soul = load_soul(name)
            self.assertTrue(soul.prose, name)
        self.assertNotIn("pick", load_soul("robot_security").autonomy)      # it watches; it moves nothing
        self.assertIn("pick", load_soul("robot_cleaning").autonomy)

    def test_a_description_becomes_a_soul_file(self) -> None:
        client = WriterClient({"role": "gardening, cheerful robot", "autonomy": ["navigate", "look", "say", "fly"],
                               "cadence_s": 5, "habits": "watering_can=garden, nonsense",
                               "prose": "I look after the plants.\n---\nI tell you when one looks dry."})
        text = asyncio.run(write_soul(client, "a gardening robot"))
        soul = parse_soul("robot_custom", text)
        self.assertEqual(soul.role, "gardening cheerful robot")            # a comma would have made it a list
        self.assertEqual(soul.autonomy, ("navigate", "look", "say"))       # no flying
        self.assertEqual(soul.cadence_s, 15.0)                             # not busier than every 15 s
        self.assertEqual(soul.habits, {"watering_can": "garden"})
        self.assertIn("I tell you when one looks dry.", soul.prose)
        self.assertNotIn("---", soul.prose)

    def test_no_description_no_soul(self) -> None:
        with self.assertRaises(ValueError):
            asyncio.run(write_soul(WriterClient({}), "   "))


if __name__ == "__main__":
    unittest.main()
