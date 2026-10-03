from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thor.world import Layout, Surface, ThorWorld  # noqa: E402
from ui.server import Session  # noqa: E402


class FakeEvent:
    def __init__(self, action_return=None) -> None:
        self.metadata = {
            "actionReturn": action_return,
            "lastActionSuccess": True,
            "errorMessage": "",
            "agent": {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "cameraHorizon": 30.0,
            },
            "objects": [],
        }
        self.third_party_camera_frames = []


class FakeController:
    def __init__(self, **options) -> None:
        self.options = options
        self.calls = []
        self.last_event = FakeEvent()

    def step(self, action: str, **options):
        self.calls.append((action, options))
        if action == "GetMapViewCameraProperties":
            return FakeEvent({
                "position": {"x": 0.0, "y": 4.0, "z": 0.0},
                "rotation": {"x": 90.0, "y": 0.0, "z": 0.0},
                "orthographic": True,
                "orthographicSize": 3.0,
                "fieldOfView": 90.0,
            })
        if action == "GetReachablePositions":
            return FakeEvent([{"x": 0.0, "y": 0.9, "z": 0.0}])
        self.last_event = FakeEvent()
        return self.last_event

    def stop(self) -> None:
        pass


class RobotPresenceTests(unittest.TestCase):
    def test_world_requests_one_visible_manipulation_agent(self) -> None:
        world = ThorWorld(width=320, height=240)
        self.addCleanup(world.close)

        with patch("ai2thor.controller.Controller", FakeController):
            world.load("FloorPlan1")

        self.assertEqual(world.controller.options.get("agentMode"), "default")
        self.assertNotIn("agentCount", world.controller.options)

    def test_layout_publishes_human_respond_point(self) -> None:
        layout = Layout(scene="FloorPlan1")
        layout.topdown = {"cx": 0.0, "cz": 0.0, "size": 3.0, "w": 640, "h": 480}
        layout.surfaces["counter_1"] = Surface(
            name="counter_1",
            thor_id="CounterTop|1",
            desc="counter 1",
            center=(0.75, -0.5),
            height=0.9,
            stand=(1.25, -0.5),
            yaw=270.0,
            horizon=30.0,
        )
        layout.user_surface = "counter_1"
        session = Session(None, "FloorPlan1", "agent", "model")
        session.layout = layout

        message = session.layout_message()

        self.assertEqual(message.get("human"), {
            "x": 1.25,
            "z": -0.5,
            "keypoint": "counter_1",
            "deliver_to_surface": "counter_1",
        })

    def test_default_agent_teleport_keeps_standing_pose(self) -> None:
        world = ThorWorld(width=320, height=240)
        self.addCleanup(world.close)

        with patch("ai2thor.controller.Controller", FakeController):
            world.load("FloorPlan1")

        teleport = next(options for action, options in world.controller.calls if action == "TeleportFull")
        self.assertIs(teleport.get("standing"), True)


if __name__ == "__main__":
    unittest.main()
