"""From a box in the head camera's picture to a place in the room, without depth.

A detector gives a 2D box. The robot knows its own camera (where it is, which way it looks, its
field of view) and its map (each surface's height and footprint), so the ray through the bottom
middle of the box, where the thing meets what it stands on, is followed down to the first
surface it lands on. That point, nudged a little further along the ray (the box's bottom edge is
the thing's near side, not its middle), is where the thing is.

Nothing here reads a simulator: a real robot has the same camera pose and the same map.

Axes are the runtime's: metres, y up, yaw = atan2(dx, dz) in degrees, horizon = tilt in degrees,
positive looking down. Pixels: u to the right, v down, from the top-left corner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

NUDGE_M = 0.04           # from the near bottom edge to the thing's middle
FOOTPRINT_MARGIN_M = 0.06
FLOOR = "floor"


@dataclass(frozen=True)
class Camera:
    x: float
    y: float                 # height of the lens
    z: float
    yaw: float
    horizon: float
    hfov: float              # degrees
    vfov: float
    w: int                   # pixels
    h: int

    def basis(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        """forward, right, up: unit vectors in the room."""
        a, t = math.radians(self.yaw), math.radians(self.horizon)
        fwd = (math.sin(a) * math.cos(t), -math.sin(t), math.cos(a) * math.cos(t))
        right = (math.cos(a), 0.0, -math.sin(a))
        up = (math.sin(a) * math.sin(t), math.cos(t), math.cos(a) * math.sin(t))
        return fwd, right, up

    def ray(self, u: float, v: float) -> tuple[float, float, float]:
        """The direction (unit) through pixel (u, v)."""
        fwd, right, up = self.basis()
        sx = (2.0 * u / self.w - 1.0) * math.tan(math.radians(self.hfov) / 2)
        sy = (1.0 - 2.0 * v / self.h) * math.tan(math.radians(self.vfov) / 2)
        d = tuple(fwd[i] + sx * right[i] + sy * up[i] for i in range(3))
        n = math.sqrt(sum(c * c for c in d))
        return (d[0] / n, d[1] / n, d[2] / n)

    def pixel(self, p: tuple[float, float, float]) -> tuple[float, float] | None:
        """Where a point in the room shows in the picture; None if it is behind the camera."""
        fwd, right, up = self.basis()
        d = (p[0] - self.x, p[1] - self.y, p[2] - self.z)
        zc = sum(d[i] * fwd[i] for i in range(3))
        if zc <= 1e-6:
            return None
        sx = sum(d[i] * right[i] for i in range(3)) / zc / math.tan(math.radians(self.hfov) / 2)
        sy = sum(d[i] * up[i] for i in range(3)) / zc / math.tan(math.radians(self.vfov) / 2)
        return (sx + 1.0) * self.w / 2.0, (1.0 - sy) * self.h / 2.0


@dataclass(frozen=True)
class Plane:
    """A surface from the map: a horizontal rectangle things can stand on."""
    name: str
    height: float
    center: tuple[float, float]       # x, z
    half: tuple[float, float]         # half-size in x and z


@dataclass(frozen=True)
class Landing:
    where: str                        # the surface's name, or "floor"
    x: float
    y: float                          # the surface's height (the thing's bottom)
    z: float
    distance_m: float                 # from the lens, along the ray


def land(cam: Camera, u: float, v: float, planes: Iterable[Plane], floor: bool = True) -> Landing | None:
    """Follow the ray through pixel (u, v) to the first surface it lands on inside that surface's
    footprint (or the floor). None if it never comes down onto anything."""
    d = cam.ray(u, v)
    best: Landing | None = None
    candidates = list(planes) + ([Plane(FLOOR, 0.0, (0.0, 0.0), (1e6, 1e6))] if floor else [])
    for p in candidates:
        if abs(d[1]) < 1e-6:
            continue
        t = (p.height - cam.y) / d[1]
        if t <= 0.05:
            continue
        x, z = cam.x + t * d[0], cam.z + t * d[2]
        if abs(x - p.center[0]) > p.half[0] + FOOTPRINT_MARGIN_M or abs(z - p.center[1]) > p.half[1] + FOOTPRINT_MARGIN_M:
            continue
        if best is None or t < best.distance_m:
            best = Landing(p.name, x, p.height, z, t)
    if best is None:
        return None
    h = math.hypot(d[0], d[2])
    if h > 1e-6:                        # from the near edge to the middle
        best = Landing(best.where, best.x + NUDGE_M * d[0] / h, best.y, best.z + NUDGE_M * d[2] / h, best.distance_m)
    return best


def box_base(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """The pixel where a thing in a box (u0, v0, u1, v1) meets what it stands on: bottom middle."""
    u0, v0, u1, v1 = box
    return (u0 + u1) / 2.0, v1
