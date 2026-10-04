"""What the robot's sensors tell it about things: the perception contract, and the stand-in.

The robot (sim/robot.py) asks its perception source, never the world, what it sees:
for perception() ten times a second, for each view of a look, and to know where to
reach. A source names things with the robot's own ids ("mug_1") and reports, for each
one in view: type, label, where (a surface, "hand", ...), x, y, z.

``StandIn`` is a perfect detector: the world's own object list, limited to what the
head camera sees, with exact labels. ``perception/vlm.py`` sees through a detector looking
at the camera's frames instead, so the runtime's memory never comes from the simulator; a
real robot runs the same source on its own camera.

A look takes its views one after another (capture) and may then detect them all at once
(detect), so a slow detector costs one wait per look, not one per view. A pick asks the
source where to close the hand (grasp): the stand-in names the world's object; a detector
gives the believed position, and the arm takes whatever is really there.
"""

from __future__ import annotations

from typing import Any, Protocol


class PerceptionSource(Protocol):
    source: str                      # stamped on everything it reports: "thor-ground-truth"
    latency_s: float                 # how much longer than a render a look's detection may take

    def objects(self) -> dict[str, dict[str, Any]]:
        """Things the head camera shows now, by id: type, label, where, x, y, z."""

    def landmarks(self) -> dict[str, dict[str, Any]]:
        """Fixed things the head camera shows now, by name: type, label, near, x, y, z."""

    async def view(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """One view of a look, taken after the head has moved: (objects, landmarks) as above.
        May take time (a model looking at the frame). The same as detect(capture())."""

    def capture(self, looking_for: str | None = None) -> Any:
        """Take this view now (the picture and the camera's pose), to detect later. looking_for: what
        the robot is searching for, in plain words, if anything; a detector may watch for it by name."""

    async def detect(self, shot: Any) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """What a captured view shows: (objects, landmarks)."""

    def knows(self, oid: str) -> bool:
        """Is this an id the robot can act on?"""

    def target(self, oid: str) -> dict[str, Any] | None:
        """Where to reach for it: where, x, y, z, and visible (in view now); None if unknown."""

    def grasp(self, oid: str) -> dict[str, Any] | None:
        """How the arm finds it: {"id": the world's name} or {"at": (x, y, z), "radius": m}."""

    def held(self, oid: str) -> None:
        """The hand closed on it."""

    def released(self, oid: str, surface: str) -> None:
        """The hand let go of it on that surface."""


class StandIn:
    """A perfect detector: the world's truth, but only what the head camera sees."""

    latency_s = 0.0

    def __init__(self, world: Any) -> None:
        self.world = world
        self.source = f"{world.source}-ground-truth"

    def objects(self) -> dict[str, dict[str, Any]]:
        return {k: o for k, o in self.world.objects().items() if o.get("visible")}

    def landmarks(self) -> dict[str, dict[str, Any]]:
        return {k: lm for k, lm in self.world.landmarks().items() if lm["visible"]}

    async def view(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        return self.objects(), self.landmarks()

    def capture(self, looking_for: str | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        return self.objects(), self.landmarks()

    async def detect(self, shot: Any) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        return shot

    def knows(self, oid: str) -> bool:
        return oid in self.world.layout.obj_ids

    def target(self, oid: str) -> dict[str, Any] | None:
        return self.world.objects().get(oid)

    def grasp(self, oid: str) -> dict[str, Any] | None:
        return {"id": oid}

    def held(self, oid: str) -> None:
        pass

    def released(self, oid: str, surface: str) -> None:
        pass


def create(world: Any) -> StandIn:
    return StandIn(world)
