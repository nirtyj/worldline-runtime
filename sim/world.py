"""The world seam: what a simulator gives the robot (sim/robot.py) and the page (ui/server.py).

ThorWorld (thor/world.py) is one. A world reached over the network is another: it
answers the reads from its latest state and makes its operations coroutines, which
``call`` awaits. The server picks one with ``--world module:factory``; the factory takes
no arguments and returns the world.

Reads are cheap and synchronous: the robot calls them from the event loop many times a
second (perception at 10 Hz, paths when planning a route), so they never wait on a
simulator. Operations that change the world go through ``call``.

A world knows the truth. Only the robot, which passes on what its sensors could know
(perception/source.py), and the page's truth panel read it; the runtime (agent/) never does.

Positions are metres on the floor plane (x, z), y up; yaw is degrees, atan2(dx, dz);
horizon is the head camera's tilt in degrees, positive looking down.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from .layout import Layout


class World(Protocol):
    source: str                      # "thor": stamped on what the robot reports
    layout: Layout | None            # the loaded home (sim/layout.py)
    frame_rev: int                   # bumps whenever the head camera may show something new
    on_slow: Callable[[str, float, float], None] | None    # (call, queued_s, ran_s), set by the server

    def scenes(self) -> list[tuple[str, str]]:
        """(name, label) for each room or house on offer; the first is the default."""

    def load(self, scene: str) -> Layout:
        """Load a room or house (run it through ``call``)."""

    def close(self) -> None:
        """Stop the simulator. Safe to call twice."""

    async def call(self, fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
        """Run one of this world's functions off the event loop, awaiting it if it's a coroutine."""

    # -- reads ----------------------------------------------------------
    def agent_pose(self) -> tuple[float, float, float, float]:
        """x, z, yaw, horizon of the robot's head camera."""

    def objects(self) -> dict[str, dict[str, Any]]:
        """Every pickupable thing by short id: type, label, where (a surface, "hand",
        "fridge", "floor", "unknown"), x, y, z, and visible (the head camera sees it now)."""

    def landmarks(self) -> dict[str, dict[str, Any]]:
        """Every landmark by name: type, label, near (a keypoint), x, y, z, visible."""

    def frame(self) -> Any:
        """The head camera's latest frame as an RGB array (None before the first load)."""

    def jpeg(self, which: str = "head") -> bytes:
        """The latest "head" or "top" (overhead) frame as JPEG; b"" if there is none."""

    def camera(self) -> dict[str, Any]:
        """The head camera itself: w, h (pixels), hfov, vfov (degrees), height (metres). The robot
        knows its own camera, so perception may use this to place what it sees."""

    def path(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]] | None:
        """Grid points from start to goal on the free floor, or None if there is no way."""

    def path_length(self, a: tuple[float, float], b: tuple[float, float]) -> float | None:
        """Metres along path(a, b), or None."""

    # -- operations (through call) -----------------------------------------
    def teleport(self, x: float, z: float, yaw: float, horizon: float) -> bool:
        """Put the robot there, facing yaw, head tilted to horizon. The reads that follow
        (objects' visible, frame, jpeg) are for the new pose."""

    def pickup(self, short: str) -> tuple[bool, str]:
        """Take that object into the hand: (ok, error)."""

    def pickup_at(self, x: float, y: float, z: float, radius: float) -> tuple[bool, str]:
        """Close the hand at that point: take whatever pickupable thing is really there (within
        radius, on the floor plane), or fail. A detector's belief aims the arm; the world decides."""

    def put(self, surface: str) -> tuple[bool, str]:
        """Put the held object down on that stretch, in front of the robot: (ok, error)."""
