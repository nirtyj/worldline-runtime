"""A made-up AI2-THOR room, for tests that need a loaded world without the simulator.

``SyntheticController`` stands in for ``ai2thor.controller.Controller`` and serves the
metadata of one small home: counters, a table, a two-level shelf, a bed reached from its
edge, a sofa too far to reach, appliances, and objects in every kind of place (on a
surface, in a bowl, in the fridge, on the floor, in the hand). ``HOUSE`` adds rooms, so
the same furniture can be loaded as a ProcTHOR house. ``snapshot`` flattens everything
the layout and the robot's read side report into plain JSON.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any


def _obj(oid: str, center: tuple[float, float, float], size: tuple[float, float, float] = (0.1, 0.1, 0.1),
         pickupable: bool = False, parents: list[str] | None = None, visible: bool = False,
         distance: float = 1.0, picked: bool = False) -> dict[str, Any]:
    cx, cy, cz = center
    return {
        "objectId": oid, "objectType": oid.split("|")[0], "pickupable": pickupable,
        "axisAlignedBoundingBox": {"center": {"x": cx, "y": cy, "z": cz},
                                   "size": {"x": size[0], "y": size[1], "z": size[2]}},
        "position": {"x": cx, "y": cy, "z": cz},
        "parentReceptacles": parents, "visible": visible, "distance": distance, "isPickedUp": picked,
    }


OBJECTS = [
    # furniture the robot can put things on
    _obj("CounterTop|1", (2.0, 0.9, 0.0), (3.0, 0.1, 0.6)),          # long: three stretches
    _obj("CounterTop|2", (4.0, 0.9, 2.0), (0.6, 0.1, 1.2)),
    _obj("DiningTable|1", (2.0, 0.75, 4.2), (1.2, 0.05, 0.8)),
    _obj("Shelf|1", (0.0, 0.5, 2.0), (0.4, 0.05, 0.8)),              # two levels of one unit
    _obj("Shelf|2", (0.0, 1.0, 2.0), (0.4, 0.05, 0.8)),
    _obj("Bed|1", (5.7, 0.5, 3.0), (2.0, 0.5, 1.6)),                 # big: stood at by its edge
    _obj("Sofa|1", (9.0, 0.4, 9.0), (2.0, 0.8, 0.9)),                # no spot near it: skipped
    # fixed things worth naming
    _obj("StoveBurner|1", (1.2, 0.95, -0.1)),
    _obj("StoveBurner|2", (1.5, 0.95, -0.1)),
    _obj("Fridge|1", (0.0, 0.9, 3.6), (0.7, 1.8, 0.7), visible=True),
    _obj("Microwave|1", (3.9, 1.2, 1.6), (0.5, 0.3, 0.4)),
    _obj("CoffeeMachine|1", (2.9, 1.0, -0.1), (0.2, 0.3, 0.2), distance=2.5),
    # things to pick up
    _obj("Apple|1", (2.9, 1.0, 0.1), pickupable=True, parents=["CounterTop|1"], visible=True),
    _obj("Apple|2", (1.0, 1.0, 0.05), pickupable=True, parents=["CounterTop|1"]),
    _obj("Bowl|1", (4.0, 1.0, 2.2), pickupable=True, parents=["CounterTop|2"], visible=True),
    _obj("Fork|1", (4.0, 1.02, 2.2), pickupable=True, parents=["Bowl|1"]),          # in the bowl: on its counter
    _obj("Mug|1", (2.1, 0.8, 4.1), pickupable=True, parents=["DiningTable|1"], distance=1.2),
    _obj("Mug|2", (2.0, 1.1, 2.0), pickupable=True, picked=True, visible=True),      # in the hand
    _obj("Book|1", (0.0, 1.05, 2.1), pickupable=True, parents=["Shelf|2"]),          # upper shelf level
    _obj("Egg|1", (0.0, 0.9, 3.6), pickupable=True, parents=["Fridge|1"]),           # inside the fridge
    _obj("Pen|1", (2.5, 0.05, 2.5), pickupable=True),                                # on the floor
    _obj("KeyChain|1", (3.0, 0.8, 3.0), pickupable=True),                            # nobody knows
    _obj("Potato|1", (2.0, 1.0, 0.0), pickupable=True, parents=["Missing|9", "CounterTop|1"]),
]
# what the head camera's segmentation shows: the mug is in the picture though THOR says "not visible"
DETECTIONS = {"Mug|1": (100, 100, 110, 110), "CoffeeMachine|1": (10, 10, 30, 30), "Pen|1": (5, 5, 8, 8)}

REACHABLE = [{"x": gx * 0.25, "y": 0.9, "z": gz * 0.25} for gx in range(2, 15) for gz in range(2, 15)
             if not (7 <= gx <= 9 and 7 <= gz <= 9)]                 # a pillar in the middle

HOUSE = {
    "rooms": [
        {"roomType": "Kitchen", "floorPolygon": [{"x": -1, "z": -1}, {"x": 5, "z": -1}, {"x": 5, "z": 2.4}, {"x": -1, "z": 2.4}]},
        {"roomType": "LivingRoom", "floorPolygon": [{"x": -1, "z": 2.4}, {"x": 5, "z": 2.4}, {"x": 5, "z": 6}, {"x": -1, "z": 6}]},
        {"roomType": "Bedroom", "floorPolygon": [{"x": 5, "z": 1}, {"x": 8, "z": 1}, {"x": 8, "z": 5}, {"x": 5, "z": 5}]},
    ],
    "metadata": {"agent": {"position": {"x": 1.0, "y": 0.9, "z": 1.0}, "rotation": {"x": 0, "y": 90, "z": 0}, "horizon": 30}},
}
HOUSE_NAME = "procthor-train-0"


class SyntheticEvent:
    def __init__(self, agent: dict[str, Any], action_return: Any = None, ok: bool = True) -> None:
        self.metadata = {"actionReturn": action_return, "lastActionSuccess": ok, "errorMessage": "",
                         "agent": agent, "objects": OBJECTS}
        self.instance_detections2D = DETECTIONS
        self.third_party_camera_frames: list[Any] = []
        self.frame = None


class SyntheticController:
    def __init__(self, **options: Any) -> None:
        self.options = options
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.agent = {"position": {"x": 1.5, "y": 0.9, "z": 1.5}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                      "cameraHorizon": 30.0}
        self.last_event = SyntheticEvent(self.agent)

    def reset(self, scene: Any = None) -> SyntheticEvent:
        self.calls.append(("reset", {"scene": scene}))
        return self.last_event

    def step(self, action: str, **options: Any) -> SyntheticEvent:
        self.calls.append((action, options))
        if action == "GetMapViewCameraProperties":
            return SyntheticEvent(self.agent, {"position": {"x": 2.0, "y": 4.0, "z": 2.0},
                                               "rotation": {"x": 90.0, "y": 0.0, "z": 0.0},
                                               "orthographic": True, "orthographicSize": 3.0, "fieldOfView": 90.0})
        if action == "GetReachablePositions":
            return SyntheticEvent(self.agent, REACHABLE)
        if action == "TeleportFull":
            self.agent = {"position": {"x": options["x"], "y": options["y"], "z": options["z"]},
                          "rotation": dict(options["rotation"]), "cameraHorizon": options["horizon"]}
        self.last_event = SyntheticEvent(self.agent)
        return self.last_event

    def stop(self) -> None:
        pass


def _plain(o: Any) -> Any:
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _plain(dataclasses.asdict(o))
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return sorted(_plain(v) for v in o)
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    return o


def snapshot(world: Any, robot: Any) -> dict[str, Any]:
    """Everything the layout and the robot's read side say about the loaded world, as JSON."""
    lay = world.layout
    out = {
        "layout": {k: _plain(getattr(lay, k)) for k in
                   ("scene", "surfaces", "start", "grid", "y", "obj_ids", "thor_ids", "landmarks", "rooms",
                    "alias", "skipped", "user_surface", "topdown")},
        "keypoints": _plain(lay.keypoints()),
        "objects": _plain(world.objects()),
        "landmarks": _plain(world.landmarks()),
        "paths": {f"{a}->{b}": _plain(world.path(a, b)) for a, b in
                  (((0.5, 0.5), (3.5, 3.5)), ((1.5, 1.5), (2.0, 2.0)), ((0.1, 0.1), (3.0, 0.5)), ((1.0, 1.0), (9.0, 9.0)))},
        "path_length": world.path_length((0.5, 0.5), (3.5, 3.5)),
        "map": _plain(robot.lookup_keypoints()),
        "perception": _plain(robot.perception()),
        "telemetry": _plain(robot.telemetry()),
        "base_state": _plain(robot.base_state()),
    }
    return json.loads(json.dumps(out))
