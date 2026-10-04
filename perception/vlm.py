"""Perception from the head camera's pictures, not from the simulator.

A detector names and boxes the things in one picture; the map places each box in the room
(project.py); the robot keeps its own names for them (entities.py). This is the seam where the
runtime stops trusting the world: its memory of what is where comes only from what a detector
said about a picture. Navigation and the arm still move in the world as they did; a pick is aimed
at where perception says the thing is (the world's pickup_at), so a wrong detection is a real miss.

    python ui/server.py --perception perception.vlm:create      # a vision model looks at each frame
    python ui/server.py --perception perception.vlm:synthetic   # boxes made from the sim's truth, with noise

A look takes all its views first and detects them together (capture, then detect). Between looks
a frame goes to the detector only when the view is new or the scene changed (brains/frame_gate.py),
at most once every MIN_INTERVAL_S, and objects() serves what the latest view of this pose found.
Landmarks and people still come from the world's own lists.

Environment:
  WORLDLINE_VLM_MODEL   the vision model (default gemini-3.8-flash; ":think" lets a Gemini model think
                        first; claude-* works too)
  WORLDLINE_VLM_LIVE    0: detect only during looks, never between them
  WORLDLINE_SYNTH       the synthetic detector's noise, e.g. "miss=0.1,confuse=0.05,jitter=6,seed=1"
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

from brains.frame_gate import FrameGate

from .entities import Entities, Entity
from .project import Camera, Plane, box_base, land

MIN_INTERVAL_S = 1.5                # between looks, at most one frame this often
SAME_VIEW_M, SAME_VIEW_DEG = 0.3, 15.0
HAND = (0.35, 1.0)                  # a held thing: this far in front of the robot, this high
GRASP_RADIUS_M = 0.3

# Not the scene's list (the robot doesn't know what's in the house): household words, so the
# detector and the robot agree on names. Anything else, the detector names itself.
HOUSEHOLD = ("mug", "cup", "glass", "bowl", "plate", "bottle", "wine_bottle", "can", "jar", "pot", "pan", "kettle",
             "spatula", "ladle", "knife", "fork", "spoon", "apple", "banana", "orange", "lemon", "tomato", "potato",
             "bread", "egg", "book", "newspaper", "magazine", "notebook", "pen", "pencil", "remote_control",
             "cell_phone", "laptop", "tablet", "keys", "key_chain", "wallet", "credit_card", "watch",
             "alarm_clock", "candle", "vase", "basket", "box", "pillow", "towel", "soap_bottle", "spray_bottle",
             "tissue_box", "toilet_paper", "plant", "teddy_bear", "basketball", "baseball_bat", "tennis_racket",
             "headphones", "sunglasses", "hat", "shoe", "statue", "figurine")

SYSTEM = (
    "You are the eyes of a home robot. In the picture from its head camera, find every small "
    "movable thing a hand could pick up: on tables, counters, shelves, sideboards, the floor. "
    "Ignore furniture, appliances, walls, doors, windows, curtains, rugs, lamps, pictures on walls, "
    "cushions, pillows and blankets, large plants in floor pots, and people. "
    "Name each thing as exactly as the picture lets you: a folded newspaper is not a book, a mug is not a vase. "
    "For each thing: type, singular snake_case, using one of these words when one fits: "
    + ", ".join(HOUSEHOLD) + "; otherwise your own word. "
    "label: a few plain words a person would say (\"red mug\", \"newspaper\"). "
    "also: other names it could be, if you are not sure (a folded paper might be a newspaper or a magazine). "
    "box_2d: [ymin, xmin, ymax, xmax] around the whole thing, each 0-1000 of the picture's height or width. "
    "confidence: 0 to 1. Report each thing once. If there is nothing, report an empty list.")
USER = "Report the movable things in this picture with report_objects."
TOOL = {
    "name": "report_objects",
    "description": "Report the movable things seen in the picture.",
    "parameters": {"type": "object", "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "properties": {"type": {"type": "string"}, "label": {"type": "string"},
                       "also": {"type": "array", "items": {"type": "string"}},
                       "box_2d": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                       "confidence": {"type": "number"}},
        "required": ["type", "box_2d"]}}}, "required": ["objects"]},
}


def type_name(raw: str) -> str:
    """"Remote Control", "remote-control", "RemoteControl" -> remote_control."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(raw).strip())
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "thing"


@dataclass
class Detection:
    type: str
    label: str
    box: tuple[float, float, float, float]       # u0, v0, u1, v1 in pixels
    confidence: float = 1.0
    also: list[str] | None = None                # other things it might be


@dataclass
class Shot:
    """One picture and everything needed to place what is in it, taken at once."""
    jpeg: bytes
    cam: Camera
    t: float
    rev: int
    landmarks: dict[str, dict[str, Any]]
    truth: Any = None                            # the synthetic detector's snapshot (privileged)
    looking_for: str | None = None               # what the robot is searching for, if anything


@dataclass
class View:
    cam: Camera
    ids: list[str]
    t: float


def parse_detections(args: dict[str, Any], w: int, h: int) -> list[Detection]:
    """A report_objects call (boxes as [ymin, xmin, ymax, xmax] in 0-1000) as pixel boxes."""
    out = []
    for o in args.get("objects") or []:
        try:
            y0, x0, y1, x1 = (float(v) for v in o["box_2d"][:4])
        except (KeyError, TypeError, ValueError):
            continue
        u0, u1 = sorted((x0 * w / 1000.0, x1 * w / 1000.0))
        v0, v1 = sorted((y0 * h / 1000.0, y1 * h / 1000.0))
        if u1 - u0 < 2 or v1 - v0 < 2:
            continue
        t = type_name(o.get("type", "thing"))
        also = [type_name(a) for a in (o.get("also") or []) if isinstance(a, str) and type_name(a) != t][:3]
        out.append(Detection(t, str(o.get("label") or t.replace("_", " ")), (u0, v0, u1, v1),
                             max(0.0, min(1.0, float(o.get("confidence", 0.8) or 0.8))), also))
    return out


class ModelDetector:
    """A vision model, asked for one forced tool call per picture."""

    def __init__(self, model: str | None = None, client: Any = None) -> None:
        from llmkit.client import make_client, provider_for
        spec = model or os.environ.get("WORLDLINE_VLM_MODEL", "gemini-3.8-flash")
        self.model, _, mode = spec.partition(":")            # "gemini-3.8-flash:think": let it think first
        self.name = spec
        kw: dict[str, Any] = {"max_tokens": 4000 if mode == "think" else 2000, "timeout": 40.0 if mode == "think" else 25.0}
        if provider_for(self.model) == "gemini":
            kw["thinking_budget"] = None if mode == "think" else 0
        self.client = client or make_client(provider_for(self.model), self.model, **kw)
        self.calls, self.tokens_in, self.ms = 0, 0, 0.0

    def snapshot(self, world: Any) -> None:
        return None

    async def detect(self, shot: Shot) -> list[Detection]:
        user = USER
        if shot.looking_for:            # open-vocabulary: told what to watch for, it names that thing so
            user += (f" The robot is looking for: {shot.looking_for}. If something in the picture could be "
                     f"that, report it with that name as its type (and what else it could be in also).")
        call = await self.client.tool_call(SYSTEM, user, [TOOL], images=[shot.jpeg])
        self.calls += 1
        self.tokens_in += call.input_tokens
        self.ms += call.latency_ms
        return parse_detections(call.args, shot.cam.w, shot.cam.h)


class SyntheticDetector:
    """Boxes made from the world's truth: each thing the camera really sees, its bounding box
    projected into the picture, then missed, mislabelled or jittered by a seeded generator. For
    tests and noise studies only: it reads the simulator, as the stand-in does."""

    CONFUSED = {"mug": "cup", "cup": "mug", "book": "notebook", "newspaper": "magazine", "apple": "tomato",
                "remote_control": "cell_phone", "vase": "bottle", "bowl": "plate", "candle": "bottle"}

    def __init__(self, miss: float = 0.0, confuse: float = 0.0, jitter: float = 0.0, seed: int = 0) -> None:
        self.miss, self.confuse, self.jitter = float(miss), float(confuse), float(jitter)
        self.rng = random.Random(int(seed))
        self.name = "synthetic" if not (miss or confuse or jitter) else f"synthetic(miss={miss},confuse={confuse},jitter={jitter})"
        self.calls = 0

    def snapshot(self, world: Any) -> dict[str, dict[str, Any]]:
        return {k: dict(o) for k, o in world.objects().items() if o.get("visible") and o.get("where") != "hand"}

    async def detect(self, shot: Shot) -> list[Detection]:
        self.calls += 1
        cam, out = shot.cam, []
        for oid in sorted(shot.truth or {}):
            o = shot.truth[oid]
            sx, sy, sz = o.get("size") or (0.12, 0.12, 0.12)
            pts = [shot.cam.pixel((o["x"] + a * sx / 2, o["y"] + b * sy / 2, o["z"] + c * sz / 2))
                   for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
            if any(p is None for p in pts) or self.rng.random() < self.miss:
                continue
            us, vs = [p[0] for p in pts], [p[1] for p in pts]
            j = (lambda: self.rng.gauss(0.0, self.jitter)) if self.jitter else (lambda: 0.0)
            u0, v0 = max(0.0, min(us) + j()), max(0.0, min(vs) + j())
            u1, v1 = min(float(cam.w), max(us) + j()), min(float(cam.h), max(vs) + j())
            if u1 - u0 < 2 or v1 - v0 < 2:
                continue
            t = o["type"]
            if self.confuse and self.rng.random() < self.confuse:
                t = self.CONFUSED.get(t, "box")
            out.append(Detection(t, t.replace("_", " "), (u0, v0, u1, v1), 0.9))
        return out


class VLMSource:
    """The perception source (perception/source.py) that sees through a detector."""

    def __init__(self, world: Any, detector: Any, live: bool = True) -> None:
        self.world = world
        self.detector = detector
        self.source = f"{world.source}-{detector.name}"
        self.live = live
        self.latency_s = 20.0 if isinstance(detector, ModelDetector) else 0.0    # extra time a look may take
        self.entities = Entities()
        self.errors: list[str] = []
        self._layout: Any = None
        self._planes: list[Plane] = []
        self._views: list[View] = []
        self._gate = FrameGate(min_interval=MIN_INTERVAL_S)
        self._gate_rev = -1
        self._bg: asyncio.Task | None = None
        self._quiet_until = 0.0                   # a look is taking its own pictures
        self._audit = getattr(world, "audit_detections", None)

    # -- the map and the camera (the robot's own knowledge, not the world's) ---
    def _fresh(self) -> None:
        lay = self.world.layout
        if lay is self._layout:
            return
        self._layout = lay
        self.entities.clear()
        self._views.clear()
        self._gate.reset()
        self._planes = [Plane(s.name, s.height, (s.center[0], s.center[1]), (s.half[0], s.half[1]))
                        for s in lay.surfaces.values()] if lay is not None else []

    def _camera(self) -> Camera:
        c = self.world.camera()
        x, z, yaw, horizon = self.world.agent_pose()
        return Camera(x, float(c["height"]), z, yaw, horizon, float(c["hfov"]), float(c["vfov"]), int(c["w"]), int(c["h"]))

    # -- a look: capture, then detect ------------------------------------------------
    def capture(self, looking_for: str | None = None) -> Shot:
        self._fresh()
        self._quiet_until = time.monotonic() + 2.0
        lms = {k: lm for k, lm in self.world.landmarks().items() if lm["visible"]}
        return Shot(self.world.jpeg("head"), self._camera(), time.monotonic(), int(self.world.frame_rev), lms,
                    self.detector.snapshot(self.world), (looking_for or "").strip() or None)

    async def detect(self, shot: Shot) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        try:
            dets = await self.detector.detect(shot)
        except Exception as e:                    # a model that is down sees nothing; say so once in a while
            self.errors.append(f"{type(e).__name__}: {e}"[:300])
            if len(self.errors) in (1, 10, 100):
                print(f"[perception] {self.detector.name} failed ({len(self.errors)}x): {self.errors[-1]}", file=sys.stderr)
            return {}, shot.landmarks
        placed = []
        hand = self._hand_pixel(shot.cam)
        for d in dets:
            if hand is not None and _covers(d.box, hand):
                continue                          # the thing in its own hand, not a thing on the furniture
            spot = land(shot.cam, *box_base(d.box), self._planes)
            if spot is None:
                continue
            placed.append({"type": d.type, "label": d.label, "x": spot.x, "y": spot.y + 0.05, "z": spot.z,
                           "where": spot.where, "confidence": d.confidence, "also": d.also or [], "box": d.box})
        ents = self.entities.assign(placed, shot.t)
        self._views.append(View(shot.cam, list(ents), shot.t))
        del self._views[:-24]
        if self._audit is not None:
            self._audit(rev=shot.rev, pose=[shot.cam.x, shot.cam.z, shot.cam.yaw, shot.cam.horizon],
                        detector=self.detector.name,
                        detections=[{"id": eid, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()
                                                   if k != "box"}, "box": [round(b) for b in p["box"]]}
                                    for eid, p in zip(ents, placed)])
        return {eid: e.as_report(True) for eid, e in ents.items()}, shot.landmarks

    async def view(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        return await self.detect(self.capture())

    # -- between looks ---------------------------------------------------------------
    def _same_view(self, a: Camera, b: Camera) -> bool:
        turn = abs((a.yaw - b.yaw + 180.0) % 360.0 - 180.0)
        return math.dist((a.x, a.z), (b.x, b.z)) <= SAME_VIEW_M and turn <= SAME_VIEW_DEG \
            and abs(a.horizon - b.horizon) <= SAME_VIEW_DEG

    def _live(self, cam: Camera) -> None:
        if not self.live or (self._bg is not None and not self._bg.done()) or time.monotonic() < self._quiet_until:
            return
        rev = int(self.world.frame_rev)
        if rev == self._gate_rev:
            return
        self._gate_rev = rev
        frame = self.world.frame()
        if frame is None:
            return
        if self._gate.decide(frame, (cam.x, cam.z, cam.yaw, cam.horizon), time.monotonic()).send:
            try:
                self._bg = asyncio.get_running_loop().create_task(self.detect(self.capture()))
            except RuntimeError:                   # no event loop (a test calling objects() directly)
                self._bg = None

    def _in_view(self, cam: Camera) -> list[str]:
        for v in reversed(self._views):
            if self._same_view(v.cam, cam):
                return v.ids
        return []

    def _hand_pixel(self, cam: Camera) -> tuple[float, float] | None:
        """Where the held thing shows in the picture, if the hand holds something."""
        if not any(e.held for e in self.entities.items.values()):
            return None
        a = math.radians(cam.yaw)
        return cam.pixel((cam.x + HAND[0] * math.sin(a), HAND[1], cam.z + HAND[0] * math.cos(a)))

    def _held_report(self, e: Entity, cam: Camera) -> dict[str, Any]:
        a = math.radians(cam.yaw)
        return {**e.as_report(True), "x": cam.x + HAND[0] * math.sin(a), "y": HAND[1], "z": cam.z + HAND[0] * math.cos(a)}

    def objects(self) -> dict[str, dict[str, Any]]:
        self._fresh()
        if self.world.layout is None:
            return {}
        cam = self._camera()
        self._live(cam)
        items = self.entities.items
        out = {eid: items[eid].as_report(True) for eid in self._in_view(cam) if eid in items and not items[eid].held}
        out.update({e.id: self._held_report(e, cam) for e in items.values() if e.held})
        return out

    def landmarks(self) -> dict[str, dict[str, Any]]:
        return {k: lm for k, lm in self.world.landmarks().items() if lm["visible"]}

    # -- acting on what it believes ------------------------------------------------------
    def knows(self, oid: str) -> bool:
        return oid in self.entities.items

    def target(self, oid: str) -> dict[str, Any] | None:
        e = self.entities.items.get(oid)
        if e is None:
            return None
        if e.held:
            return self._held_report(e, self._camera())
        return e.as_report(oid in self._in_view(self._camera()))

    def grasp(self, oid: str) -> dict[str, Any] | None:
        """Where the arm should close: the believed position, not the simulator's id."""
        e = self.entities.items.get(oid)
        return None if e is None else {"at": (e.x, e.y, e.z), "radius": GRASP_RADIUS_M}

    def held(self, oid: str) -> None:
        if oid in self.entities.items:
            self.entities.items[oid].held = True

    def released(self, oid: str, surface: str) -> None:
        e = self.entities.items.get(oid)
        if e is None:
            return
        e.held = False
        lay = self.world.layout
        s = lay.surfaces.get(surface) if lay is not None else None
        if s is not None:                       # down in front of the robot on that stretch; a look will tell
            e.x, e.z, e.y, e.where = s.center[0], s.center[1], s.height + 0.05, surface


def _covers(box: tuple[float, float, float, float], p: tuple[float, float], grow: float = 0.25) -> bool:
    u0, v0, u1, v1 = box
    gu, gv = (u1 - u0) * grow, (v1 - v0) * grow
    return u0 - gu <= p[0] <= u1 + gu and v0 - gv <= p[1] <= v1 + gv


def _opts(spec: str) -> dict[str, float]:
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        k, _, v = part.partition("=")
        out[k.strip()] = float(v)
    return out


def create(world: Any) -> VLMSource:
    """A vision model looks at the head camera's frames (WORLDLINE_VLM_MODEL)."""
    return VLMSource(world, ModelDetector(), live=os.environ.get("WORLDLINE_VLM_LIVE", "1") != "0")


def synthetic(world: Any) -> VLMSource:
    """Boxes from the simulator's truth, with the noise in WORLDLINE_SYNTH (tests, noise studies)."""
    return VLMSource(world, SyntheticDetector(**_opts(os.environ.get("WORLDLINE_SYNTH", ""))),
                     live=os.environ.get("WORLDLINE_VLM_LIVE", "1") != "0")
