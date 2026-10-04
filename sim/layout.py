"""The layout: a home turned into the names the runtime uses.

A world (thor/world.py, or any other simulator) describes its furniture, its objects
and its free floor as plain records; the functions here turn them into a ``Layout``:
surfaces and their keypoints, short object ids, landmarks, and paths on the free grid.
Nothing here knows which simulator the records came from.

Naming, so a model can read it:
  surfaces   one per reachable stretch of furniture: "counter_2b" is the second
             half of the second counter top. Each surface has one keypoint of
             the same name, where the robot stands facing it. In a house the
             room comes first: "kitchen_counter_1a", "bedroom_bed_1". Shelf
             levels stacked in one unit share a single surface.
  keypoints  the surface spots, plus "start" (where the robot begins).
  objects    pickupable things, "apple_1", "butter_knife_1".
  landmarks  fixed things worth naming, "microwave_1", "stove_1" (all burners
             as one), "fridge_1". They can't be picked up; they're how a person
             describes places ("next to the toaster").

Positions are metres on the floor plane (x, z), y up; yaw is degrees, atan2(dx, dz).
"""

from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

LANDMARK_SEEN_M = 3.0    # appliances are big: recognisable from further away
GRID = 0.25
SEGMENT_M = 1.3          # a surface longer than this gets one keypoint per stretch
MAX_STAND_OFF = 1.6      # a spot farther than this from its stretch is useless
EDGE_STAND_OFF = 1.25    # ...unless it is this close to the furniture's edge (beds, big tables)
REACH_M = 1.5            # the arm reaches this far from the robot's centre
MAX_REACH_HEIGHT = 1.5
NAV_SPEED = 0.6          # m/s, as in the old sim


def place_points(half: tuple[float, float]) -> list[tuple[float, float]]:
    """Centre of a stretch first, then rings of points inside it."""
    hx, hz = max(half[0] - 0.08, 0.0), max(half[1] - 0.08, 0.0)
    pts = [(0.0, 0.0)]
    for r in (0.15, 0.3, 0.45):
        for fx, fz in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            dx, dz = max(-hx, min(hx, fx * r)), max(-hz, min(hz, fz * r))
            if (dx, dz) not in pts:
                pts.append((dx, dz))
    return pts


def _edge_dist(p: tuple[float, float], centre: tuple[float, float], half: tuple[float, float]) -> float:
    """Distance from a floor point to a furniture footprint (an axis-aligned rectangle)."""
    dx = max(abs(p[0] - centre[0]) - half[0], 0.0)
    dz = max(abs(p[1] - centre[1]) - half[1], 0.0)
    return math.hypot(dx, dz)


def snake(name: str) -> str:
    """CreditCard -> credit_card, TVStand -> tv_stand."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def words(name: str) -> str:
    return snake(name).replace("_", " ")


@dataclass
class Thing:
    """One physical thing, as a world reports it when a home is loaded."""
    id: str                                   # the world's own id for it
    type: str                                 # snake_case: "counter_top", "apple"
    center: tuple[float, float, float]        # of its bounding box: x, y, z
    size: tuple[float, float, float]          # of its bounding box
    position: tuple[float, float, float]      # where the world says it is
    pickupable: bool = False
    surface: str | None = None                # what people call it, if the robot can put things on it: "counter"
    landmark: str | None = None               # what people call it, if it's a fixed thing worth naming: "stove"
    grouped: bool = False                     # several of these make one landmark (a stove's burners)


@dataclass
class Placed:
    """Where one thing is right now, as a world reports it."""
    type: str                                 # snake_case
    position: tuple[float, float, float]      # x, y, z
    held: bool = False                        # in the robot's hand
    parents: tuple[str, ...] = ()             # world ids of what it rests on or is inside
    is_surface: bool = False                  # furniture the robot can put things on
    carries: bool = False                     # a bowl, a plate: what's in it is "on" whatever it's on


@dataclass
class Surface:
    name: str
    thor_id: str                # the world's id of the furniture this stretch belongs to
    desc: str
    center: tuple[float, float]  # x, z of the stretch
    height: float
    stand: tuple[float, float]  # x, z of its keypoint
    yaw: float                  # facing the stretch
    horizon: float              # camera tilt when standing there
    half: tuple[float, float] = (0.3, 0.3)   # half-size of the stretch in x and z


@dataclass
class Landmark:
    name: str
    label: str
    thor_ids: list[str]         # the world's ids: a stove is several burners
    center: tuple[float, float, float]   # x, y, z
    near: str                   # the keypoint to stand at to use it


@dataclass
class Layout:
    scene: str
    surfaces: dict[str, Surface] = field(default_factory=dict)
    start: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)   # x, z, yaw, horizon
    grid: set[tuple[int, int]] = field(default_factory=set)
    y: float = 0.9               # the agent's height; teleports need it
    obj_ids: dict[str, str] = field(default_factory=dict)      # short -> the world's id
    thor_ids: dict[str, str] = field(default_factory=dict)     # the world's id -> short
    landmarks: dict[str, "Landmark"] = field(default_factory=dict)
    rooms: dict[str, dict[str, Any]] = field(default_factory=dict)   # houses only
    alias: dict[str, str] = field(default_factory=dict)      # the world's id of a stacked shelf -> its surface
    skipped: list[tuple[str, str]] = field(default_factory=list)   # surfaces with no spot, and why
    user_surface: str | None = None
    topdown: dict[str, Any] = field(default_factory=dict)      # the overhead camera, for the page
    grid_step: float = GRID      # metres between the grid's cells

    def keypoints(self) -> dict[str, tuple[float, float, float, float]]:
        kps = {"start": self.start}
        for s in self.surfaces.values():
            kps[s.name] = (s.stand[0], s.stand[1], s.yaw, s.horizon)
        return kps


# ----------------------------------------------------------------------
# Rooms
# ----------------------------------------------------------------------
def room_at(rooms_: dict[str, dict[str, Any]], x: float, z: float) -> str | None:
    """Which room's floor polygon contains (x, z)? Nearest room centre if none does."""
    for name, r in rooms_.items():
        if _inside(r["polygon"], x, z):
            return name
    if not rooms_:
        return None
    return min(rooms_, key=lambda n: (rooms_[n]["center"][0] - x) ** 2 + (rooms_[n]["center"][1] - z) ** 2)


def _inside(poly: list[tuple[float, float]], x: float, z: float) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, zi = poly[i]
        xj, zj = poly[j]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


# ----------------------------------------------------------------------
# Naming a loaded home
# ----------------------------------------------------------------------
def layout_surfaces(lay: Layout, things: list[Thing]) -> None:
    """Split the furniture into stretches and give each a name and a spot to stand at."""
    counts: dict[tuple[str | None, str], int] = collections.Counter()
    used: list[tuple[float, float]] = []
    stands = [(gx * lay.grid_step, gz * lay.grid_step) for gx, gz in lay.grid]
    # in a house, stand in the same room as the furniture, never behind a wall
    stand_room = {p: room_at(lay.rooms, *p) for p in stands} if lay.rooms else {}
    lay.skipped = []
    # Shelf levels stacked in one unit: keep the lowest as the surface, fold the rest into it.
    surf = [t for t in things if t.surface is not None]
    centre = {t.id: (t.center[0], t.center[2]) for t in surf}
    type_of = {t.id: t.type for t in things}
    primary: dict[str, str] = {}
    for t in sorted(surf, key=lambda t: t.center[1]):
        below = next((q for q in primary.values() if type_of[q] == t.type
                      and math.dist(centre[q], centre[t.id]) < 0.35), None)
        primary[t.id] = below or t.id
    for t in sorted(things, key=lambda t: t.id):
        kind = t.surface
        if kind is None or primary[t.id] != t.id:
            continue
        cx, cz = t.center[0], t.center[2]
        ex, ez = t.size[0], t.size[2]
        height = t.center[1] + t.size[1] / 2
        n = max(1, math.ceil(max(ex, ez) / SEGMENT_M))
        along_x = ex >= ez
        room = room_at(lay.rooms, cx, cz) if lay.rooms else None
        counts[(room, kind)] += 1
        num = counts[(room, kind)]
        base = f"{t.type if kind != 'counter' else 'counter'}_{num}"
        if room:
            base = f"{room}_{base}"
        for i in range(n):
            f = (i + 0.5) / n - 0.5
            seg = (cx + f * ex, cz) if along_x else (cx, cz + f * ez)
            free = [p for p in stands if all(math.dist(p, u) > 0.3 for u in used)
                    and (not room or stand_room.get(p) == room)]
            if not free:
                lay.skipped.append((t.id, "no free spot in the room"))
                continue
            best = min(free, key=lambda p: math.dist(p, seg))
            if math.dist(best, seg) > MAX_STAND_OFF:
                # big furniture: its centre is far from everywhere; stand by its nearest edge
                half = (ex / n / 2, ez / 2) if along_x else (ex / 2, ez / n / 2)
                best = min(free, key=lambda p: _edge_dist(p, seg, half))
                if _edge_dist(best, seg, half) > EDGE_STAND_OFF:
                    lay.skipped.append((t.id, f"nearest spot {math.dist(best, seg):.1f} m away"))
                    continue
            used.append(best)
            name = base + ("" if n == 1 else "abcdefgh"[i])
            yaw = math.degrees(math.atan2(seg[0] - best[0], seg[1] - best[1])) % 360
            horizon = 45.0 if height < 0.6 else 30.0 if height < 1.1 else 10.0
            part = "" if n == 1 else f", part {'abcdefgh'[i]}"
            where = f" in the {lay.rooms[room]['label']}" if room else ""
            lay.surfaces[name] = Surface(name, t.id, f"{kind} {num}{part}{where}", seg,
                                         round(height, 2), best, round(yaw, 1), horizon,
                                         (ex / n / 2, ez / 2) if along_x else (ex / 2, ez / n / 2))

    for tid, first in primary.items():
        if tid != first:
            home = next((x.name for x in lay.surfaces.values() if x.thor_id == first), None)
            if home:
                lay.alias[tid] = home


def name_objects(lay: Layout, things: list[Thing]) -> None:
    counts: dict[str, int] = collections.Counter()
    for t in sorted(things, key=lambda t: t.id):
        if not t.pickupable:
            continue
        counts[t.type] += 1
        short = f"{t.type}_{counts[t.type]}"
        lay.obj_ids[short] = t.id
        lay.thor_ids[t.id] = short


def name_landmarks(lay: Layout, things: list[Thing]) -> None:
    groups: dict[str, list[list[Thing]]] = collections.defaultdict(list)
    for t in sorted(things, key=lambda t: t.id):
        if t.landmark is None or t.pickupable:
            continue
        if t.grouped and groups[t.type]:
            groups[t.type][0].append(t)
        else:
            groups[t.type].append([t])
    kps = lay.keypoints()
    for members in groups.values():
        for i, group in enumerate(members, 1):
            x = sum(t.position[0] for t in group) / len(group)
            y = sum(t.position[1] for t in group) / len(group)
            z = sum(t.position[2] for t in group) / len(group)
            near = min(kps, key=lambda k: math.dist(kps[k][:2], (x, z)))
            label = group[0].landmark
            name = f"{snake(label).replace(' ', '_')}_{i}"
            lay.landmarks[name] = Landmark(name, label, [t.id for t in group], (x, y, z), near)


# ----------------------------------------------------------------------
# Where a thing is, in the layout's names
# ----------------------------------------------------------------------
def where_of(lay: Layout, o: Placed, lookup: Callable[[str], Placed | None], depth: int = 0) -> str:
    """A surface name, "hand", what it is inside ("fridge"), "floor" or "unknown".
    ``lookup`` gives the same record for whatever it rests on, by the world's id."""
    if o.held:
        return "hand"
    for pid in o.parents:
        p = lookup(pid)
        if p is None:
            continue
        if p.is_surface:
            return nearest_stretch(lay, pid, o.position[0], o.position[2]) or lay.alias.get(pid) or p.type
        if p.carries and depth < 2:
            return where_of(lay, p, lookup, depth + 1)
        return p.type                         # inside a fridge, a cabinet, a drawer...
    return "floor" if o.position[1] < 0.2 else "unknown"


def nearest_stretch(lay: Layout, thor_id: str, x: float, z: float) -> str | None:
    segs = [s for s in lay.surfaces.values() if s.thor_id == thor_id]
    if not segs:
        return None
    return min(segs, key=lambda s: math.dist(s.center, (x, z))).name


# ----------------------------------------------------------------------
# Paths on the free grid
# ----------------------------------------------------------------------
def grid_path(grid: set[tuple[int, int]], step: float, start: tuple[float, float],
              goal: tuple[float, float]) -> list[tuple[float, float]] | None:
    s = (round(start[0] / step), round(start[1] / step))
    g = (round(goal[0] / step), round(goal[1] / step))
    if s not in grid:
        s = min(grid, key=lambda p: (p[0] - s[0]) ** 2 + (p[1] - s[1]) ** 2)
    prev: dict[tuple[int, int], tuple[int, int] | None] = {s: None}
    queue = collections.deque([s])
    while queue:
        cur = queue.popleft()
        if cur == g:
            break
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cur[0] + dx, cur[1] + dz)
            if nxt in grid and nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    if g not in prev:
        return None
    out = []
    node: tuple[int, int] | None = g
    while node is not None:
        out.append((node[0] * step, node[1] * step))
        node = prev[node]
    return out[::-1]


def path_length(grid: set[tuple[int, int]], step: float, a: tuple[float, float],
                b: tuple[float, float]) -> float | None:
    p = grid_path(grid, step, a, b)
    return None if p is None else round((len(p) - 1) * step, 2)
