"""What the robot's sensors tell it about things: the perception contract, and the stand-in.

The robot (sim/robot.py) asks its perception source, never the world, what it sees:
for perception() ten times a second, for each view of a look, and to know where to
reach. A source names things with the robot's own ids ("mug_1") and reports, for each
one in view: type, label, where (a surface, "hand", ...), x, y, z.

``StandIn`` is a perfect detector: the world's own object list, limited to what the
head camera sees, with exact labels. A real robot, or a source that runs a model on
the camera frames, implements the same calls.
"""

from __future__ import annotations

from typing import Any, Protocol


class PerceptionSource(Protocol):
    source: str                      # stamped on everything it reports: "thor-ground-truth"

    def objects(self) -> dict[str, dict[str, Any]]:
        """Things the head camera shows now, by id: type, label, where, x, y, z."""

    def landmarks(self) -> dict[str, dict[str, Any]]:
        """Fixed things the head camera shows now, by name: type, label, near, x, y, z."""

    async def view(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """One view of a look, taken after the head has moved: (objects, landmarks) as above.
        May take time (a model looking at the frame)."""

    def knows(self, oid: str) -> bool:
        """Is this an id the robot can act on?"""

    def target(self, oid: str) -> dict[str, Any] | None:
        """Where to reach for it: where, x, y, z, and visible (in view now); None if unknown."""


class StandIn:
    """A perfect detector: the world's truth, but only what the head camera sees."""

    def __init__(self, world: Any) -> None:
        self.world = world
        self.source = f"{world.source}-ground-truth"

    def objects(self) -> dict[str, dict[str, Any]]:
        return {k: o for k, o in self.world.objects().items() if o.get("visible")}

    def landmarks(self) -> dict[str, dict[str, Any]]:
        return {k: lm for k, lm in self.world.landmarks().items() if lm["visible"]}

    async def view(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        return self.objects(), self.landmarks()

    def knows(self, oid: str) -> bool:
        return oid in self.world.layout.obj_ids

    def target(self, oid: str) -> dict[str, Any] | None:
        return self.world.objects().get(oid)


def create(world: Any) -> StandIn:
    return StandIn(world)
