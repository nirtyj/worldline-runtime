"""Perception through a detector (perception/vlm.py): boxes placed with the map, the robot's own
names, and a pick aimed at a position. No model and no simulator: a stub world with one table,
and the synthetic detector (boxes from the stub's truth)."""

from __future__ import annotations

import asyncio
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception.entities import Entities  # noqa: E402
from perception.project import Camera, Plane, box_base, land  # noqa: E402
from perception.vlm import SyntheticDetector, VLMSource, parse_detections, type_name  # noqa: E402
from sim.layout import Layout, Surface  # noqa: E402

CAM = dict(hfov=106.26, vfov=90.0, w=640, h=480)


class Projection(unittest.TestCase):
    def test_a_point_on_a_table_comes_back_from_its_pixel(self) -> None:
        table = Plane("table_1", 0.75, (0.6, 1.4), (0.5, 0.4))
        for yaw, tilt in ((0.0, 30.0), (25.0, 40.0), (340.0, 20.0)):
            cam = Camera(0.2, 1.4, 0.1, yaw, tilt, **CAM)
            p = (0.75, 0.75, 1.3)
            u, v = cam.pixel(p)
            hit = land(cam, u, v, [table])
            self.assertEqual(hit.where, "table_1", (yaw, tilt))
            self.assertLess(math.dist((hit.x, hit.z), (p[0], p[2])), 0.06, (yaw, tilt, hit))   # the nudge, at most

    def test_off_the_table_is_the_floor(self) -> None:
        cam = Camera(0.0, 1.4, 0.0, 0.0, 45.0, **CAM)
        u, v = cam.pixel((1.5, 0.0, 1.0))
        self.assertEqual(land(cam, u, v, [Plane("table_1", 0.75, (0.0, 1.0), (0.3, 0.3))]).where, "floor")

    def test_right_of_centre_is_to_the_robots_right(self) -> None:
        cam = Camera(0.0, 1.4, 0.0, 0.0, 0.0, **CAM)          # facing +z: right is +x
        self.assertGreater(cam.pixel((1.0, 1.4, 2.0))[0], 320)
        self.assertEqual(box_base((10, 20, 30, 60)), (20.0, 60))


class Names(unittest.TestCase):
    def test_close_is_the_same_thing_far_is_another(self) -> None:
        ents = Entities()
        a = ents.assign([{"type": "mug", "x": 1.0, "y": 0.8, "z": 1.0, "where": "t"}], 0.0)
        b = ents.assign([{"type": "mug", "x": 1.1, "y": 0.8, "z": 1.05, "where": "t"},
                         {"type": "mug", "x": 3.0, "y": 0.8, "z": 1.0, "where": "u"},
                         {"type": "book", "x": 1.0, "y": 0.8, "z": 1.0, "where": "t"}], 1.0)
        self.assertEqual(list(a), ["mug_1"])
        self.assertEqual(sorted(b), ["book_1", "mug_1", "mug_2"])
        self.assertAlmostEqual(ents.items["mug_1"].x, 1.1)

    def test_called_something_else_at_the_same_spot_is_still_that_thing(self) -> None:
        ents = Entities()
        ents.assign([{"type": "magazine", "x": 0.1, "y": 0.8, "z": 1.5, "where": "t", "also": ["newspaper"]}], 0.0)
        got = ents.assign([{"type": "newspaper", "x": 0.15, "y": 0.8, "z": 1.55, "where": "t"}], 1.0)
        ents.assign([{"type": "magazine", "x": 0.12, "y": 0.8, "z": 1.5, "where": "t"}], 2.0)
        self.assertEqual(list(got), ["magazine_1"])
        self.assertEqual(len(ents.items), 1)
        self.assertEqual(ents.items["magazine_1"].label, "magazine (maybe newspaper)")

    def test_model_boxes_and_names(self) -> None:
        dets = parse_detections({"objects": [{"type": "Remote Control", "box_2d": [500, 250, 750, 500], "confidence": 0.7},
                                             {"type": "mug", "box_2d": [1, 2]}]}, 640, 480)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].type, "remote_control")
        self.assertEqual(dets[0].box, (160.0, 240.0, 320.0, 360.0))
        self.assertEqual(type_name("CellPhone"), "cell_phone")


class StubWorld:
    """One table, one mug on it, a robot looking at it. objects() is the truth."""
    source = "stub"

    def __init__(self) -> None:
        s = Surface("table_1", "fix:table", "table", (1.0, 1.6), 0.75, (1.0, 0.6), 0.0, 30.0, (0.5, 0.4))
        self.layout = Layout(scene="stub", surfaces={"table_1": s})
        self.pose = (1.0, 0.6, 0.0, 30.0)
        self.mug = {"type": "mug", "label": "mug", "where": "table_1", "x": 1.1, "y": 0.8, "z": 1.6,
                    "visible": True, "size": (0.1, 0.1, 0.1)}
        self.frame_rev = 1
        self.picked: list = []

    def camera(self):
        return {**CAM, "height": 1.4}

    def agent_pose(self):
        return self.pose

    def objects(self):
        return {"mug_1": dict(self.mug)}

    def landmarks(self):
        return {}

    def jpeg(self, which="head"):
        return b""

    def frame(self):
        return None

    def pickup_at(self, x, y, z, radius):
        self.picked.append((x, y, z))
        return math.dist((x, z), (self.mug["x"], self.mug["z"])) <= radius, ""


class ThroughADetector(unittest.TestCase):
    def test_a_look_names_and_places_what_the_detector_saw(self) -> None:
        world = StubWorld()
        src = VLMSource(world, SyntheticDetector(), live=False)
        objs, _ = asyncio.run(src.detect(src.capture()))
        self.assertEqual(list(objs), ["mug_1"])
        m = objs["mug_1"]
        self.assertEqual(m["where"], "table_1")
        self.assertLess(math.dist((m["x"], m["z"]), (1.1, 1.6)), 0.15)
        self.assertTrue(src.knows("mug_1"))
        self.assertIn("mug_1", src.objects())                 # still in view: the same pose
        world.pose = (1.0, 0.6, 90.0, 30.0)                    # turned away: not in view any more
        self.assertNotIn("mug_1", src.objects())
        self.assertFalse(src.target("mug_1")["visible"])

        spec = src.grasp("mug_1")                              # the arm is aimed at the belief
        ok, _ = world.pickup_at(*spec["at"], spec["radius"])
        self.assertTrue(ok)
        src.held("mug_1")
        self.assertEqual(src.objects()["mug_1"]["where"], "hand")
        src.released("mug_1", "table_1")
        self.assertEqual(src.target("mug_1")["where"], "table_1")

    def test_the_thing_in_its_own_hand_is_not_seen_on_the_furniture(self) -> None:
        world = StubWorld()
        src = VLMSource(world, SyntheticDetector(), live=False)
        asyncio.run(src.detect(src.capture()))
        src.held("mug_1")
        cam = src._camera()
        hu, hv = src._hand_pixel(cam)

        class HandInView:                       # a detector that sees only the held mug
            name = "hand"

            def snapshot(self, world):
                return None

            async def detect(self, shot):
                from perception.vlm import Detection
                return [Detection("mug", "mug", (hu - 20, hv - 20, hu + 20, hv + 20))]
        src.detector = HandInView()
        objs, _ = asyncio.run(src.detect(src.capture()))
        self.assertEqual(objs, {})
        self.assertEqual(sorted(src.entities.items), ["mug_1"])

    def test_a_missed_thing_is_not_known(self) -> None:
        world = StubWorld()
        src = VLMSource(world, SyntheticDetector(miss=1.0), live=False)
        objs, _ = asyncio.run(src.detect(src.capture()))
        self.assertEqual(objs, {})
        self.assertFalse(src.knows("mug_1"))


if __name__ == "__main__":
    unittest.main()
