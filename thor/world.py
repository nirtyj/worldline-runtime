"""AI2-THOR as the robot's world: an iTHOR room or a multi-room ProcTHOR house
(thor/procthor.py), its layout turned into the names the runtime uses
(rooms, keypoints, surfaces, short object ids), and frames from the robot's
head camera and an overhead camera.

Every controller call blocks and THOR is not thread-safe, so all of them run
on one worker thread; ``await world.call(fn, ...)`` hops onto it.

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
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import io
import math
import os
import re
import signal
import subprocess
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image

from . import procthor

CONNECT_TIMEOUT_S = 30.0       # a fresh simulator that hasn't connected by then never will
_fifo_servers: "weakref.WeakSet[Any]" = weakref.WeakSet()


def _track_fifo_servers() -> None:
    """Remember every ai2thor FIFO server, so a connect that never happens can be broken
    (ThorWorld._unstick). ai2thor builds the server itself, so this wraps its __init__."""
    from ai2thor.fifo_server import FifoServer
    if getattr(FifoServer, "_runtime_tracked", False):
        return
    init = FifoServer.__init__

    def tracked(self: Any, *a: Any, **kw: Any) -> None:
        init(self, *a, **kw)
        _fifo_servers.add(self)
    FifoServer.__init__ = tracked
    FifoServer._runtime_tracked = True

SURFACE_TYPES = {
    "CounterTop": "counter", "DiningTable": "dining table", "CoffeeTable": "coffee table",
    "SideTable": "side table", "Desk": "desk", "Dresser": "dresser", "Shelf": "shelf",
    "TVStand": "TV stand", "Sofa": "sofa", "Bed": "bed", "ArmChair": "armchair",
    "Ottoman": "ottoman", "SinkBasin": "sink",
}
LANDMARK_TYPES = {                      # THOR type -> what people call it
    "Microwave": "microwave", "Toaster": "toaster", "StoveBurner": "stove",
    "CoffeeMachine": "coffee machine", "Fridge": "fridge", "GarbageCan": "bin",
    "Television": "TV", "Toilet": "toilet", "Bathtub": "bathtub", "ShowerHead": "shower",
    "FloorLamp": "floor lamp", "HousePlant": "plant",
    "LaundryHamper": "laundry hamper", "Safe": "safe", "Desktop": "computer",
}
GROUPED_LANDMARKS = {"StoveBurner"}     # several THOR objects, one name
MIN_PIXELS = 40          # a box this big in the head camera counts as seen
LANDMARK_SEEN_M = 3.0    # appliances are big: recognisable from further away
HELD_CONTAINERS = {"Bowl", "Plate", "Mug", "Cup", "Pan", "Pot"}   # an apple in a bowl is "on" the bowl's surface
GRID = 0.25
SEGMENT_M = 1.3          # a surface longer than this gets one keypoint per stretch
MAX_STAND_OFF = 1.6      # a spot farther than this from its stretch is useless
EDGE_STAND_OFF = 1.25    # ...unless it is this close to the furniture's edge (beds, big tables)
REACH_M = 1.5            # the arm reaches this far from the robot's centre
MAX_REACH_HEIGHT = 1.5
NAV_SPEED = 0.6          # m/s, as in the old sim

SCENES = (
    procthor.HOUSES +
    [(f"FloorPlan{i}", f"Kitchen {i}") for i in (1, 2, 3, 5, 10, 11)] +
    [(f"FloorPlan{i}", f"Living room {i - 200}") for i in (201, 203, 209, 212)] +
    [(f"FloorPlan{i}", f"Bedroom {i - 300}") for i in (301, 302, 311, 320)] +
    [(f"FloorPlan{i}", f"Bathroom {i - 400}") for i in (401, 403)]
)


def _place_points(half: tuple[float, float]) -> list[tuple[float, float]]:
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


def by_type(objs: list[dict[str, Any]], thor_id: str) -> str | None:
    return next((o["objectType"] for o in objs if o["objectId"] == thor_id), None)


def snake(name: str) -> str:
    """CreditCard -> credit_card, TVStand -> tv_stand."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def words(name: str) -> str:
    return snake(name).replace("_", " ")


@dataclass
class Surface:
    name: str
    thor_id: str                # the receptacle this stretch belongs to
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
    thor_ids: list[str]         # a stove is several burners
    center: tuple[float, float, float]   # x, y, z
    near: str                   # the keypoint to stand at to use it


@dataclass
class Layout:
    scene: str
    surfaces: dict[str, Surface] = field(default_factory=dict)
    start: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)   # x, z, yaw, horizon
    grid: set[tuple[int, int]] = field(default_factory=set)
    y: float = 0.9               # the agent's height; teleports need it
    obj_ids: dict[str, str] = field(default_factory=dict)      # short -> THOR id
    thor_ids: dict[str, str] = field(default_factory=dict)     # THOR id -> short
    landmarks: dict[str, "Landmark"] = field(default_factory=dict)
    rooms: dict[str, dict[str, Any]] = field(default_factory=dict)   # ProcTHOR houses only
    alias: dict[str, str] = field(default_factory=dict)      # THOR id of a stacked shelf -> its surface
    skipped: list[tuple[str, str]] = field(default_factory=list)   # surfaces with no spot, and why
    user_surface: str | None = None
    topdown: dict[str, Any] = field(default_factory=dict)      # the overhead camera, for the page

    def keypoints(self) -> dict[str, tuple[float, float, float, float]]:
        kps = {"start": self.start}
        for s in self.surfaces.values():
            kps[s.name] = (s.stand[0], s.stand[1], s.yaw, s.horizon)
        return kps


class ThorWorld:
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width, self.height = width, height
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="thor")
        self.controller: Any = None
        self._stuck_fds: list[int] = []          # pipe ends opened by _unstick, closed before the retry
        self.event: Any = None
        self.layout: Layout | None = None
        self.frame_rev = 0
        self.held: str | None = None      # THOR id of the object in the hand
        self.on_slow: Callable[[str, float, float], None] | None = None   # (call, queued_s, ran_s)

    # ------------------------------------------------------------------
    async def call(self, fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
        loop = asyncio.get_running_loop()
        queued = time.monotonic()
        timing: dict[str, float] = {}

        def run() -> Any:
            timing["start"] = time.monotonic()
            try:
                return fn(*args, **kw)
            finally:
                timing["end"] = time.monotonic()

        try:
            return await loop.run_in_executor(self._pool, run)
        finally:
            total = time.monotonic() - queued
            if total > 0.5 and self.on_slow is not None and "end" in timing:
                self.on_slow(getattr(fn, "__name__", "call"), round(timing["start"] - queued, 2),
                             round(timing["end"] - timing["start"], 2))

    def close(self) -> None:
        self._unstick()                          # a simulator still connecting: stop it too
        self._stop_simulator()
        for fd in self._stuck_fds:
            os.close(fd)
        self._stuck_fds.clear()
        self._pool.shutdown(wait=False)

    # ------------------------------------------------------------------
    # The simulator process: one per world, always reaped
    # ------------------------------------------------------------------
    def _unity_proc(self) -> subprocess.Popen | None:
        return getattr(getattr(self.controller, "server", None), "unity_proc", None)

    def alive(self) -> bool:
        """Is there a simulator behind this world, and is it still running?"""
        proc = self._unity_proc()
        return self.controller is not None and proc is not None and proc.poll() is None

    def _stop_simulator(self) -> None:
        """Stop Unity and reap it. AI2-THOR's own stop can leave a zombie (or, if it
        raises, a live process), so kill whatever is left and wait for it."""
        controller, proc = self.controller, self._unity_proc()
        self.controller, self.event, self.layout = None, None, None
        if controller is None:
            return
        try:
            controller.stop()
        except Exception:
            pass
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[thor] simulator {proc.pid} did not exit after SIGKILL", flush=True)

    @staticmethod
    def reap_orphans() -> list[int]:
        """Kill AI2-THOR simulators left behind by a process that died without stopping
        them (their parent is now launchd, pid 1). Run once at server start."""
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
        killed = []
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) == 3 and parts[1] == "1" and "/.ai2thor/releases/thor-" in parts[2]:
                try:
                    os.kill(int(parts[0]), signal.SIGKILL)
                    killed.append(int(parts[0]))
                except OSError:
                    pass
        return killed

    # ------------------------------------------------------------------
    # Worker-thread functions (call them through ``call``)
    # ------------------------------------------------------------------
    def _unstick(self) -> None:
        """A fresh simulator sometimes never opens its end of the pipe, and ai2thor waits for
        it in a plain open() that no timeout covers (fifo_server.py _recv_message). Kill that
        simulator and open the pipe's other end ourselves: the open() returns, the first read
        gets nothing and times out, and load() retries on a new simulator."""
        for srv in list(_fifo_servers):
            if getattr(srv, "server_pipe", True) is not None:
                continue                                  # connected (or not a waiting server)
            print(f"[thor] the simulator did not connect in {CONNECT_TIMEOUT_S:.0f} s; stopping it", flush=True)
            proc = getattr(srv, "unity_proc", None)
            if proc is not None and proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            try:                                          # kept open until the retry, so the read waits, not spins
                self._stuck_fds.append(os.open(srv.server_pipe_path, os.O_WRONLY | os.O_NONBLOCK))
            except OSError:
                pass

    def load(self, scene: str) -> Layout:
        """Load a room or house. A simulator that died since the last load is replaced
        first. Unity sometimes never finishes initialising a fresh scene; when a load
        times out, the simulator is stopped, reaped, restarted, and the load tried once more."""
        if self.controller is not None and not self.alive():
            print("[thor] the simulator has exited; starting a new one", flush=True)
            self._stop_simulator()
        for attempt in (1, 2):
            try:
                return self._load(scene)
            except (TimeoutError, BrokenPipeError, ConnectionError) as e:
                self._stop_simulator()
                for fd in self._stuck_fds:
                    os.close(fd)
                self._stuck_fds.clear()
                if attempt == 2:
                    raise
                print(f"[thor] loading {scene} failed ({type(e).__name__}); restarting the simulator", flush=True)
        raise AssertionError("unreachable")

    def _load(self, scene: str) -> Layout:
        from ai2thor.controller import Controller
        house = procthor.load_house(scene) if procthor.is_house(scene) else None
        if self.controller is None:
            _track_fifo_servers()
            watchdog = threading.Timer(CONNECT_TIMEOUT_S, self._unstick)
            watchdog.daemon = True
            watchdog.start()
            try:
                self.controller = Controller(scene="Procedural" if house else scene, agentMode="default",
                                             width=self.width, height=self.height,
                                             gridSize=GRID, visibilityDistance=REACH_M, fieldOfView=90,
                                             renderInstanceSegmentation=True,   # "seen" = has pixels in the frame
                                             server_timeout=45.0)               # a hung load retries sooner
            finally:
                watchdog.cancel()
        if house is not None:
            self.controller.reset(scene=house)
            a = house["metadata"]["agent"]               # where the house says the robot starts
            self._step(action="TeleportFull", x=a["position"]["x"], y=a["position"]["y"], z=a["position"]["z"],
                       rotation=a["rotation"], horizon=a["horizon"], standing=True)
        elif self.event is not None:
            self.controller.reset(scene=scene)
        c = self.controller
        props = c.step(action="GetMapViewCameraProperties").metadata["actionReturn"]
        self._step(action="AddThirdPartyCamera", **props, skyboxColor="white")
        reach = c.step(action="GetReachablePositions").metadata["actionReturn"]
        agent = self.event.metadata["agent"]
        lay = Layout(scene=scene)
        lay.y = agent["position"]["y"]
        lay.grid = {(round(p["x"] / GRID), round(p["z"] / GRID)) for p in reach}
        sx, sz = agent["position"]["x"], agent["position"]["z"]
        lay.start = (sx, sz, agent["rotation"]["y"], 30.0)
        lay.topdown = {"cx": props["position"]["x"], "cz": props["position"]["z"],
                       "size": props["orthographicSize"], "w": self.width, "h": self.height}
        if house is not None:
            lay.rooms = procthor.rooms(house)
        self._layout_surfaces(lay, self.event.metadata["objects"])
        self._name_objects(lay, self.event.metadata["objects"])
        self._name_landmarks(lay, self.event.metadata["objects"])
        if lay.surfaces:
            lay.user_surface = min(lay.surfaces.values(),
                                   key=lambda s: math.dist(s.stand, (sx, sz))).name
        self.layout = lay
        self.held = None
        self.teleport(*lay.start)
        return lay

    def _layout_surfaces(self, lay: Layout, objs: list[dict[str, Any]]) -> None:
        counts: dict[tuple[str | None, str], int] = collections.Counter()
        used: list[tuple[float, float]] = []
        stands = [(gx * GRID, gz * GRID) for gx, gz in lay.grid]
        # in a house, stand in the same room as the furniture, never behind a wall
        stand_room = {p: procthor.room_at(lay.rooms, *p) for p in stands} if lay.rooms else {}
        lay.skipped = []
        # Shelf levels stacked in one unit: keep the lowest as the surface, fold the rest into it.
        surf = [o for o in objs if o["objectType"] in SURFACE_TYPES]
        centre = {o["objectId"]: (o["axisAlignedBoundingBox"]["center"]["x"], o["axisAlignedBoundingBox"]["center"]["z"])
                  for o in surf}
        primary: dict[str, str] = {}
        for o in sorted(surf, key=lambda o: o["axisAlignedBoundingBox"]["center"]["y"]):
            below = next((q for q in primary.values() if by_type(objs, q) == o["objectType"]
                          and math.dist(centre[q], centre[o["objectId"]]) < 0.35), None)
            primary[o["objectId"]] = below or o["objectId"]
        for o in sorted(objs, key=lambda o: o["objectId"]):
            kind = SURFACE_TYPES.get(o["objectType"])
            if kind is None or primary[o["objectId"]] != o["objectId"]:
                continue
            box = o["axisAlignedBoundingBox"]
            cx, cz = box["center"]["x"], box["center"]["z"]
            ex, ez = box["size"]["x"], box["size"]["z"]
            height = box["center"]["y"] + box["size"]["y"] / 2
            n = max(1, math.ceil(max(ex, ez) / SEGMENT_M))
            along_x = ex >= ez
            room = procthor.room_at(lay.rooms, cx, cz) if lay.rooms else None
            counts[(room, kind)] += 1
            num = counts[(room, kind)]
            base = f"{snake(o['objectType']) if kind != 'counter' else 'counter'}_{num}"
            if room:
                base = f"{room}_{base}"
            for i in range(n):
                f = (i + 0.5) / n - 0.5
                seg = (cx + f * ex, cz) if along_x else (cx, cz + f * ez)
                free = [p for p in stands if all(math.dist(p, u) > 0.3 for u in used)
                        and (not room or stand_room.get(p) == room)]
                if not free:
                    lay.skipped.append((o["objectId"], "no free spot in the room"))
                    continue
                best = min(free, key=lambda p: math.dist(p, seg))
                if math.dist(best, seg) > MAX_STAND_OFF:
                    # big furniture: its centre is far from everywhere; stand by its nearest edge
                    half = (ex / n / 2, ez / 2) if along_x else (ex / 2, ez / n / 2)
                    best = min(free, key=lambda p: _edge_dist(p, seg, half))
                    if _edge_dist(best, seg, half) > EDGE_STAND_OFF:
                        lay.skipped.append((o["objectId"], f"nearest spot {math.dist(best, seg):.1f} m away"))
                        continue
                used.append(best)
                name = base + ("" if n == 1 else "abcdefgh"[i])
                yaw = math.degrees(math.atan2(seg[0] - best[0], seg[1] - best[1])) % 360
                horizon = 45.0 if height < 0.6 else 30.0 if height < 1.1 else 10.0
                part = "" if n == 1 else f", part {'abcdefgh'[i]}"
                where = f" in the {lay.rooms[room]['label']}" if room else ""
                lay.surfaces[name] = Surface(name, o["objectId"], f"{kind} {num}{part}{where}", seg,
                                             round(height, 2), best, round(yaw, 1), horizon,
                                             (ex / n / 2, ez / 2) if along_x else (ex / 2, ez / n / 2))

        for tid, first in primary.items():
            if tid != first:
                home = next((x.name for x in lay.surfaces.values() if x.thor_id == first), None)
                if home:
                    lay.alias[tid] = home

    def _name_objects(self, lay: Layout, objs: list[dict[str, Any]]) -> None:
        counts: dict[str, int] = collections.Counter()
        for o in sorted(objs, key=lambda o: o["objectId"]):
            if not o["pickupable"]:
                continue
            t = snake(o["objectType"])
            counts[t] += 1
            short = f"{t}_{counts[t]}"
            lay.obj_ids[short] = o["objectId"]
            lay.thor_ids[o["objectId"]] = short

    def _name_landmarks(self, lay: Layout, objs: list[dict[str, Any]]) -> None:
        groups: dict[str, list[list[dict[str, Any]]]] = collections.defaultdict(list)
        for o in sorted(objs, key=lambda o: o["objectId"]):
            t = o["objectType"]
            if t not in LANDMARK_TYPES or o["pickupable"]:
                continue
            if t in GROUPED_LANDMARKS and groups[t]:
                groups[t][0].append(o)
            else:
                groups[t].append([o])
        kps = lay.keypoints()
        for t, members in groups.items():
            for i, group in enumerate(members, 1):
                x = sum(o["position"]["x"] for o in group) / len(group)
                y = sum(o["position"]["y"] for o in group) / len(group)
                z = sum(o["position"]["z"] for o in group) / len(group)
                near = min(kps, key=lambda k: math.dist(kps[k][:2], (x, z)))
                name = f"{snake(LANDMARK_TYPES[t]).replace(' ', '_')}_{i}"
                lay.landmarks[name] = Landmark(name, LANDMARK_TYPES[t], [o["objectId"] for o in group],
                                               (x, y, z), near)

    def _step(self, **action: Any) -> Any:
        self.event = self.controller.step(**action)
        self.frame_rev += 1
        return self.event

    def teleport(self, x: float, z: float, yaw: float, horizon: float) -> bool:
        ev = self._step(action="TeleportFull", x=x, y=self.layout.y if self.layout else 0.9, z=z,
                        rotation=dict(x=0, y=yaw, z=0), horizon=horizon, standing=True)
        return bool(ev.metadata["lastActionSuccess"])

    def pickup(self, short: str) -> tuple[bool, str]:
        ev = self._step(action="PickupObject", objectId=self.layout.obj_ids[short], forceAction=True)
        ok = bool(ev.metadata["lastActionSuccess"])
        if ok:
            self.held = self.layout.obj_ids[short]
        return ok, ev.metadata.get("errorMessage") or ""

    def put(self, surface: str) -> tuple[bool, str]:
        """Put the held object down on this stretch, in front of the robot.

        THOR's PutObject drops it anywhere on the whole piece of furniture (the
        far end of a table counts), so first try points on this stretch and
        keep the first one that really lands here."""
        s = self.layout.surfaces[surface]
        held = self.held
        short = self.layout.thor_ids.get(held or "")
        for dx, dz in _place_points(s.half):
            ev = self._step(action="PlaceObjectAtPoint", objectId=held,
                            position=dict(x=s.center[0] + dx, y=s.height + 0.02, z=s.center[1] + dz))
            if not ev.metadata["lastActionSuccess"]:
                continue
            if short and self.objects()[short]["where"] == surface:
                self.held = None
                return True, ""
            self._step(action="PickupObject", objectId=held, forceAction=True)   # landed elsewhere: take it back
        ev = self._step(action="PutObject", objectId=s.thor_id, forceAction=True, placeStationary=True)
        ok = bool(ev.metadata["lastActionSuccess"])
        if ok:
            self.held = None
        return ok, ev.metadata.get("errorMessage") or ""

    def jpeg(self, which: str = "head", quality: int = 72) -> bytes:
        if which == "head":
            arr = self.event.frame
        else:
            frames = self.event.third_party_camera_frames
            if not frames:
                return b""
            arr = frames[0]
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Reading the current state (cheap; safe from the event loop)
    # ------------------------------------------------------------------
    def agent_pose(self) -> tuple[float, float, float, float]:
        a = self.event.metadata["agent"]
        return a["position"]["x"], a["position"]["z"], a["rotation"]["y"], a["cameraHorizon"]

    def _seen(self, o: dict[str, Any], dets: dict[str, Any], max_m: float) -> bool:
        """Would a camera see it? THOR's visible flag, or enough pixels in the frame.

        THOR's flag casts rays to a few points on the object, so a thin fork
        behind a plant's leaves can be "not visible" while it is plainly in the
        picture; the instance segmentation counts the pixels themselves."""
        if o.get("visible"):
            return True
        box = dets.get(o["objectId"])
        return (box is not None and (box[2] - box[0]) * (box[3] - box[1]) >= MIN_PIXELS
                and o.get("distance", 99.0) <= max_m)

    def objects(self) -> dict[str, dict[str, Any]]:
        """Ground truth for every pickupable object: where it is, in our names.
        "visible" means the head camera sees it now (see _seen)."""
        lay = self.layout
        by_id = {o["objectId"]: o for o in self.event.metadata["objects"]}
        dets = self.event.instance_detections2D or {}
        out = {}
        for short, tid in lay.obj_ids.items():
            o = by_id.get(tid)
            if o is None:
                continue
            out[short] = {"type": snake(o["objectType"]), "label": words(o["objectType"]),
                          "where": self._where(o, by_id), "x": o["position"]["x"], "z": o["position"]["z"],
                          "y": o["position"]["y"], "visible": self._seen(o, dets, REACH_M)}
        return out

    def landmarks(self) -> dict[str, dict[str, Any]]:
        """Ground truth for the fixed things: where they are and whether the camera sees them."""
        by_id = {o["objectId"]: o for o in self.event.metadata["objects"]}
        dets = self.event.instance_detections2D or {}
        out = {}
        for name, lm in self.layout.landmarks.items():
            parts = [by_id[t] for t in lm.thor_ids if t in by_id]
            out[name] = {"type": snake(lm.label).replace(" ", "_"), "label": lm.label, "near": lm.near,
                         "x": lm.center[0], "y": lm.center[1], "z": lm.center[2],
                         "visible": any(self._seen(p, dets, LANDMARK_SEEN_M) for p in parts)}
        return out

    def _where(self, o: dict[str, Any], by_id: dict[str, dict[str, Any]], depth: int = 0) -> str:
        if o.get("isPickedUp"):
            return "hand"
        for pid in o.get("parentReceptacles") or []:
            p = by_id.get(pid)
            if p is None:
                continue
            if p["objectType"] in SURFACE_TYPES:
                return (self._nearest_stretch(pid, o["position"]["x"], o["position"]["z"])
                        or self.layout.alias.get(pid) or snake(p["objectType"]))
            if p["objectType"] in HELD_CONTAINERS and depth < 2:
                return self._where(p, by_id, depth + 1)
            return snake(p["objectType"])        # inside a fridge, a cabinet, a drawer...
        return "floor" if o["position"]["y"] < 0.2 else "unknown"

    def _nearest_stretch(self, thor_id: str, x: float, z: float) -> str | None:
        segs = [s for s in self.layout.surfaces.values() if s.thor_id == thor_id]
        if not segs:
            return None
        return min(segs, key=lambda s: math.dist(s.center, (x, z))).name

    # ------------------------------------------------------------------
    # Paths on the reachable grid
    # ------------------------------------------------------------------
    def path(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]] | None:
        grid = self.layout.grid
        s = (round(start[0] / GRID), round(start[1] / GRID))
        g = (round(goal[0] / GRID), round(goal[1] / GRID))
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
            out.append((node[0] * GRID, node[1] * GRID))
            node = prev[node]
        return out[::-1]

    def path_length(self, a: tuple[float, float], b: tuple[float, float]) -> float | None:
        p = self.path(a, b)
        return None if p is None else round((len(p) - 1) * GRID, 2)
