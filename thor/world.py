"""AI2-THOR as the robot's world (the World seam in sim/world.py): an iTHOR room or a
multi-room ProcTHOR house (thor/procthor.py), its layout turned into the names the
runtime uses by sim/layout.py (rooms, keypoints, surfaces, short object ids), and frames
from the robot's head camera and an overhead camera.

Every controller call blocks and THOR is not thread-safe, so all of them run
on one worker thread; ``await world.call(fn, ...)`` hops onto it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import math
import os
import signal
import subprocess
import threading
import time
import weakref
from typing import Any, Callable

from PIL import Image

from sim.layout import (GRID, LANDMARK_SEEN_M, MAX_REACH_HEIGHT, NAV_SPEED, REACH_M, Landmark,  # noqa: F401
                        Layout, Placed, Surface, Thing, grid_path, layout_surfaces, name_landmarks,
                        name_objects, place_points, snake, where_of, words)
from sim.layout import path_length as grid_path_length

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
HELD_CONTAINERS = {"Bowl", "Plate", "Mug", "Cup", "Pan", "Pot"}   # an apple in a bowl is "on" the bowl's surface

SCENES = (
    procthor.HOUSES +
    [(f"FloorPlan{i}", f"Kitchen {i}") for i in (1, 2, 3, 5, 10, 11)] +
    [(f"FloorPlan{i}", f"Living room {i - 200}") for i in (201, 203, 209, 212)] +
    [(f"FloorPlan{i}", f"Bedroom {i - 300}") for i in (301, 302, 311, 320)] +
    [(f"FloorPlan{i}", f"Bathroom {i - 400}") for i in (401, 403)]
)


def _things(objs: list[dict[str, Any]]) -> list[Thing]:
    """THOR's object metadata as the plain records sim/layout.py names."""
    out = []
    for o in objs:
        box, t = o["axisAlignedBoundingBox"], o["objectType"]
        out.append(Thing(
            id=o["objectId"], type=snake(t),
            center=(box["center"]["x"], box["center"]["y"], box["center"]["z"]),
            size=(box["size"]["x"], box["size"]["y"], box["size"]["z"]),
            position=(o["position"]["x"], o["position"]["y"], o["position"]["z"]),
            pickupable=bool(o["pickupable"]), surface=SURFACE_TYPES.get(t),
            landmark=LANDMARK_TYPES.get(t), grouped=t in GROUPED_LANDMARKS))
    return out


def _placed(o: dict[str, Any]) -> Placed:
    """Where one THOR object is, as sim/layout.py's where_of reads it."""
    return Placed(type=snake(o["objectType"]),
                  position=(o["position"]["x"], o["position"]["y"], o["position"]["z"]),
                  held=bool(o.get("isPickedUp")), parents=tuple(o.get("parentReceptacles") or ()),
                  is_surface=o["objectType"] in SURFACE_TYPES, carries=o["objectType"] in HELD_CONTAINERS)


class ThorWorld:
    source = "thor"                     # stamped on what the robot reports

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

    def scenes(self) -> list[tuple[str, str]]:
        """The rooms and houses on offer: (name, label) for the page's room menu."""
        return list(SCENES)

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
        things = _things(self.event.metadata["objects"])
        layout_surfaces(lay, things)
        name_objects(lay, things)
        name_landmarks(lay, things)
        if lay.surfaces:
            lay.user_surface = min(lay.surfaces.values(),
                                   key=lambda s: math.dist(s.stand, (sx, sz))).name
        self.layout = lay
        self.held = None
        self.teleport(*lay.start)
        return lay

    def _step(self, **action: Any) -> Any:
        self.event = self.controller.step(**action)
        self.frame_rev += 1
        return self.event

    def teleport(self, x: float, z: float, yaw: float, horizon: float) -> bool:
        ev = self._step(action="TeleportFull", x=x, y=self.layout.y if self.layout else 0.9, z=z,
                        rotation=dict(x=0, y=yaw, z=0), horizon=horizon, standing=True)
        return bool(ev.metadata["lastActionSuccess"])

    def pickup_at(self, x: float, y: float, z: float, radius: float = 0.3) -> tuple[bool, str]:
        """Close the hand at that point: the nearest pickupable thing within radius, or nothing."""
        near = [(math.dist((o["x"], o["z"]), (x, z)), short) for short, o in self.objects().items()
                if o["where"] != "hand" and abs(o["y"] - y) < 0.6]
        if not near or min(near)[0] > radius:
            return False, "the hand closed on nothing"
        return self.pickup(min(near)[1])

    def camera(self) -> dict[str, Any]:
        """The head camera: THOR's field of view (90 degrees) is vertical."""
        h, w = self.event.frame.shape[:2]
        cam = self.event.metadata.get("cameraPosition") or {}
        return {"w": w, "h": h, "vfov": 90.0, "hfov": math.degrees(2 * math.atan(math.tan(math.radians(45.0)) * w / h)),
                "height": float(cam.get("y", self.layout.y + 0.675 if self.layout else 1.575))}

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
        for dx, dz in place_points(s.half):
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
    def frame(self) -> Any:
        """The head camera's latest frame, an RGB array (None before the first load)."""
        return None if self.event is None else self.event.frame

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

        def lookup(pid: str) -> Placed | None:
            return _placed(by_id[pid]) if pid in by_id else None
        out = {}
        for short, tid in lay.obj_ids.items():
            o = by_id.get(tid)
            if o is None:
                continue
            box = (o.get("axisAlignedBoundingBox") or {}).get("size") or {}
            out[short] = {"type": snake(o["objectType"]), "label": words(o["objectType"]),
                          "where": where_of(lay, _placed(o), lookup), "x": o["position"]["x"], "z": o["position"]["z"],
                          "y": o["position"]["y"], "visible": self._seen(o, dets, REACH_M),
                          "size": (box.get("x", 0.12), box.get("y", 0.12), box.get("z", 0.12))}
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

    # ------------------------------------------------------------------
    # Paths on the reachable grid
    # ------------------------------------------------------------------
    def path(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]] | None:
        return grid_path(self.layout.grid, self.layout.grid_step, start, goal)

    def path_length(self, a: tuple[float, float], b: tuple[float, float]) -> float | None:
        return grid_path_length(self.layout.grid, self.layout.grid_step, a, b)
