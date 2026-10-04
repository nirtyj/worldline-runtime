#!/usr/bin/env python3
"""Worldline playground: chat with a robot in an AI2-THOR house and watch the runtime work.

    .venv-thor/bin/python ui/server.py        # then open http://localhost:8765

Another world plugs in with --world module:factory (sim/world.py is the contract), and
another way of seeing with --perception module:factory (perception/source.py).

Keys come from .env: GEMINI_API_KEY (the planner, Gemini 3.8 Flash) and, only if
you pick Claude in the model menu, ANTHROPIC_API_KEY.

    chat ──(text)──► InteractiveUser.next() ──► Runtime (agent/) ──► CompositeBrain ──► planner
                                                   │   ▲                (classify, next step)
    AI2-THOR room ◄── SimRobot (nav, arms, say) ◄──┘   └── perception + body sense (fused state)
         │
         └── head + overhead camera ──► the page

The page is an observer: it reads THOR's ground truth, which the runtime never
can. That is what lets you see belief and truth disagree.
"""

from __future__ import annotations

import argparse
import atexit
import signal
import asyncio
import base64
import http
import importlib
import itertools
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from websockets.asyncio.server import broadcast, serve  # noqa: E402
from websockets.datastructures import Headers  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402
from websockets.http11 import Response  # noqa: E402

from agent.episodes import load_episodes, summarize  # noqa: E402
from agent.harness import is_stop  # noqa: E402
from agent.model import REFERENCE_DEFAULT_MODEL  # noqa: E402
from agent.soul import SoulPersona, load_soul, parse_soul, write_soul  # noqa: E402
from llmkit.brain import client_from_options  # noqa: E402
from agent.memory import usual_place, volatility  # noqa: E402
from brains.composite import CompositeBrain  # noqa: E402
from brains.frame_gate import FrameGate  # noqa: E402
from brains.interface import BrainInfo  # noqa: E402
from sim.clock import SimClock  # noqa: E402
from sim.log import EventLog  # noqa: E402
from sim.robot import SimRobot  # noqa: E402
from sim.world import World  # noqa: E402
from ui.interactive_user import InteractiveUser  # noqa: E402
from ui.recorder import CallRecorder  # noqa: E402
from ui.robot_map import RobotMap  # noqa: E402

INDEX = Path(__file__).resolve().parent / "index.html"
LOG_DIR = ROOT / "runs" / "playground"
TICK_S = 0.2
HEAD_EVERY_S = 0.2
TOP_EVERY_S = 1.0
PLANNER = "agent.model:create_brain"
# The world plugs in here: "module:factory", where factory() returns a World (sim/world.py).
WORLD = "thor:create_world"
# How the robot sees: "module:factory", where factory(world) returns a perception source
# (perception/source.py). Unset: the stand-in, the world's own object list limited to the camera's view.
PERCEPTION: str | None = None
# System 1 plugs in here: "module:factory", where factory(on_status) returns an object with
# `status`, and async run(), route(text), update(context), frame(jpeg, where), robot_said(text)
# and observe() (see brains/interface.py System1). Unset: the planner classifies every message.
SYSTEM1 = os.environ.get("SYSTEM1")
S1_ROUTE_TIMEOUT_S = 2.0      # longer than this and the planner classifies instead
S1_IMAGES: dict[str, bytes] = {}      # the frames System 1's observer was shown, served at /s1/<id>.jpg
S1_IMAGES_KEEP = 400                  # the newest ones (about 10 MB)
_s1_image_ids = itertools.count(1)
S1_OBSERVE_EVERY_S = 5.0      # ask for observations this often while new frames arrive
S1_OWN_GRACE_S = 2.0          # frames this soon after a pick or place still show the robot's own doing
QUIET_EVENTS = {"speech_ended"}
MEMORY_ROWS = {"memory_saved", "place_learned", "note_saved", "delivered", "recall", "persona_goal_end", "observation"}
VIEWS = ROOT / "ui" / "views"

AGENTS = [
    ("agent", "Runtime (agent/)"),
    ("baseline.agent", "Naive baseline"),
    ("agent.mutants:forget_cancelled_grasp", "Mutant: forgets cancelled grasp"),
    ("agent.mutants:cancel_on_everything", "Mutant: cancels on everything"),
    ("agent.mutants:no_wait_for_chunk", "Mutant: doesn't wait for chunk"),
    ("agent.mutants:no_drop_speech", "Mutant: keeps stale speech"),
    ("agent.mutants:trust_success", "Mutant: trusts success flag"),
    ("agent.mutants:no_keyword_stop", "Mutant: stop goes through model"),
    ("agent.mutants:no_stale_check", "Mutant: no stale-decision check"),
]
MODELS = [
    ("gemini-3.8-flash", "Gemini 3.8 Flash (fast)"),
    ("gemini-3.8-flash:think", "Gemini 3.8 Flash (thinking)"),
    ("claude-sonnet-5", "Sonnet 5"),
    ("claude-haiku-4-5-20251001", "Haiku 4.5 (faster, cheaper)"),
]
SOULS = [                                     # the robot's soul (personas/<name>.md): the page's Soul menu, or --soul
    ("robot", "Helper"),
    ("robot_chatty", "Chatty"),
    ("robot_cleaning", "Cleaning"),
    ("robot_security", "Security"),
    ("custom", "Custom…"),                   # written on the page from a description, or --soul-file
]
CUSTOM_SOUL_MAX = 8000                        # characters


def make_soul(name: str, text: str | None) -> Any:
    """A soul from the menu: personas/<name>.md, or the custom one written on the page."""
    if name == "custom":
        if not text:
            raise ValueError("write the custom soul first (Soul → Custom…)")
        return parse_soul("robot_custom", text)
    return load_soul(name)


def keep_image(jpeg: bytes) -> str:
    """Keep a frame System 1 was shown, for the page's model-call drawer; returns its URL."""
    key = str(next(_s1_image_ids))
    S1_IMAGES[key] = jpeg
    while len(S1_IMAGES) > S1_IMAGES_KEEP:
        S1_IMAGES.pop(next(iter(S1_IMAGES)))
    return f"/s1/{key}.jpg"


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env into the environment (existing values win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip().removeprefix("export ").strip(), value.strip().strip("'\""))


def _import(spec: str) -> Any:
    """"package.module:attr" -> the attribute (or the module when there's no ":attr")."""
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr) if attr else module


def _jsonable(o: Any) -> Any:
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def dumps(msg: dict[str, Any]) -> str:
    return json.dumps(msg, default=_jsonable)


class Session:
    """One room, one runtime, one brain, fed by the chat box."""

    def __init__(self, world: World, scene: str, agent: str, model: str,
                 soul: str = "robot", speed: float = 1.0, soul_text: str | None = None) -> None:
        self.world, self.scene, self.agent, self.model = world, scene, agent, model
        self.soul, self.speed, self.soul_text = soul, speed, soul_text
        self.soul_client: Any = None
        self.error: str | None = None
        self.runtime: Any = None
        self.planner: Any = None
        self.task: asyncio.Task | None = None
        self.recorder: CallRecorder | None = None
        self.calls_log: Path | None = None
        self._ev_i = 0
        self._tr_i = 0
        self._call_rev = 0
        self._bg: list[asyncio.Task] = []

    async def start(self) -> None:
        self.clock = SimClock(self.speed)        # real models run in wall time; slower is for talking over it
        self.log = EventLog(self.clock)
        self.layout = await self.world.call(self.world.load, self.scene)
        self.robot_map = RobotMap(set(self.layout.grid), self.layout.grid_step)   # what the robot itself knows of the space
        self.world.on_slow = lambda name, queued, ran: self.log.emit(
            "slow_sim_call", call=name, queued_s=queued, ran_s=ran)
        perception = _import(PERCEPTION)(self.world) if PERCEPTION else None
        self.robot = SimRobot(self.world, self.clock, self.log, perception)
        self.user = InteractiveUser(self.clock, self.log)
        self.map = self.robot.lookup_keypoints()
        model, _, mode = self.model.partition(":")
        options = {"model": model, "temperature": "none"}
        if model.startswith("gemini"):
            # Gemini thinks by default; "fast" turns that off (about 1-2 s a step)
            options["thinking_budget"] = "none" if mode == "think" else "0"
        info = BrainInfo(scenario_id=self.scene, map=self.map, meanings=None, options=options, clock=self.clock)
        try:
            self.planner = _import(PLANNER)(info)
        except Exception as e:
            self.error = f"The planner could not start: {e}"
            return
        self.calls_log = LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{self.scene}_model_calls.txt"
        self.recorder = CallRecorder(self.planner, self.clock, self._version, self.calls_log)
        brain = CompositeBrain(self.planner)
        spec = self.agent if ":" in self.agent else self.agent + ":create_runtime"
        try:
            self.runtime = _import(spec)(self.robot, self.user, brain, self.clock)
            if callable(getattr(self.runtime, "use_persona", None)):
                # the robot's soul: personas/<soul>.md, read by its own LLM; quiet until the page says otherwise
                self.soul_client = client_from_options(options, REFERENCE_DEFAULT_MODEL)
                self.runtime.use_persona(SoulPersona(make_soul(self.soul, self.soul_text), self.soul_client,
                                                     name="robot", level="off"))
        except Exception:
            self.error = "The runtime could not start:\n" + traceback.format_exc()
            return
        self.task = asyncio.create_task(self.runtime.run())

    async def stop(self) -> None:
        for t in self._bg:
            t.cancel()
        if self.task:
            self.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.task), 3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        if hasattr(self, "robot"):
            await self.robot.shutdown()

    def _version(self) -> int:
        rt = self.runtime
        if rt is None:
            return 0
        task = getattr(rt, "task", None)
        return task.intent_version if task is not None else getattr(rt, "version", 0)

    def set_soul(self, name: str, level: str, text: str | None = None) -> None:
        """Swap the robot's soul mid-session; the runtime drops whatever the old one was doing."""
        soul = make_soul(name, text)
        self.soul, self.soul_text = name, text
        if self.soul_client is not None and callable(getattr(self.runtime, "use_persona", None)):
            self.runtime.use_persona(SoulPersona(soul, self.soul_client, name="robot", level=level))

    def hear(self, text: str, directive: dict[str, Any] | None = None) -> None:
        if self.runtime is None:
            raise RuntimeError(self.error or "nothing is running")
        self.user.say(text, directive)

    def record_system1(self, purpose: str, input_text: str, response: Any, latency_s: float | None,
                       images: list[dict[str, Any]] | None = None) -> None:
        """System 1's calls go in the page's model-call list next to the planner's."""
        if self.recorder is None:
            return
        now = round(self.clock.now(), 2)
        self.recorder.record_external({
            "via": "system1", "purpose": purpose, "t_start": round(now - (latency_s or 0), 2), "t_end": now,
            "version_start": self._version(), "version_end": self._version(), "input": input_text,
            "response": {"tool": purpose, "args": response, "text": ""}, "latency_s": latency_s,
            "status": "ok" if response else "fallback", **({"images": images} if images else {})})

    # ------------------------------------------------------------------
    # What the page sees
    # ------------------------------------------------------------------
    def _trace_rows(self) -> list[dict[str, Any]]:
        rt = self.runtime
        if rt is None:
            return []
        rows = getattr(getattr(rt, "tracer", None), "rows", None)
        if rows is None:
            rows = getattr(rt, "_log", None) or []
        return rows

    def _events_from(self, start: int) -> list[dict[str, Any]]:
        return [{**e, "i": i} for i, e in enumerate(self.log.events[start:], start)
                if e["type"] not in QUIET_EVENTS]

    def _trace_from(self, start: int) -> list[dict[str, Any]]:
        return [{**r, "i": i} for i, r in enumerate(self._trace_rows()[start:], start)]

    def layout_message(self) -> dict[str, Any]:
        lay = self.layout
        human = None
        if lay.user_surface:
            x, z, _, _ = lay.keypoints()[lay.user_surface]
            human = {"x": x, "z": z, "keypoint": lay.user_surface,
                     "deliver_to_surface": lay.user_surface}
        return {
            "scene": lay.scene, "topdown": lay.topdown,
            "keypoints": {k: {"x": v[0], "z": v[1], "yaw": v[2]} for k, v in lay.keypoints().items()},
            "surfaces": {s.name: {"desc": s.desc, "x": s.center[0], "z": s.center[1], "height": s.height}
                         for s in lay.surfaces.values()},
            "user_surface": lay.user_surface, "human": human,
            "grid": sorted([list(c) for c in lay.grid]), "grid_step": lay.grid_step,   # the nav stack's free cells
            "rooms": {k: {"label": r["label"], "x": r["center"][0], "z": r["center"][1],
                          "polygon": [list(p) for p in r.get("polygon") or []]} for k, r in lay.rooms.items()},
        }

    def init_message(self) -> dict[str, Any]:
        return {"type": "init", "config": {"scene": self.scene, "agent": self.agent, "model": self.model,
                                           "soul": self.soul, "speed": self.clock.speed, "paused": self.clock.paused},
                "map": self.map, "layout": self.layout_message(),
                "events": self._events_from(0), "trace": self._trace_from(0),
                "calls": self.recorder.calls if self.recorder else [],
                "calls_log": str(self.calls_log.relative_to(ROOT)) if self.calls_log else None,
                "error": self.error, **self.state(), "robot_map": self.robot_map.full()}

    def memory_message(self) -> dict[str, Any]:
        """What the robot has stored, for the page's Memory tab: spatial memory,
        notes, earlier sessions and the procedural graph. Sent on change, not every frame."""
        rt = self.runtime
        mem = getattr(rt, "memory", None)
        if mem is None:
            return {"type": "memory", "scene": self.scene, "available": False}
        wall = time.time()
        room_of = {k: v.get("room") for k, v in (self.map.get("keypoints") or {}).items()}
        spatial = []
        for oid, m in sorted(mem.data["objects"].items()):
            hist = m.get("history") or []
            spatial.append({
                "id": oid, "type": m.get("type"), "surface": m.get("surface"), "room": room_of.get(m.get("surface")),
                "status": m.get("status", "seen"), "usual": usual_place(hist),
                "volatility": volatility(m.get("type", ""), hist), "seen_ago_s": round(wall - m.get("seen_wall", wall)),
                "history": [{"surface": h["surface"], "room": room_of.get(h["surface"]), "ago_s": round(wall - h["wall"])}
                            for h in hist][::-1]})
        landmarks = [{"id": k, "label": m.get("label"), "near": m.get("near"), "room": room_of.get(m.get("near"))}
                     for k, m in sorted(mem.data["landmarks"].items())]
        past = getattr(getattr(rt, "recaller", None), "past", None) or []
        sessions = [summarize(rows) for rows in past[-9:]]
        sessions.append({**summarize(rt.tracer.rows), "wall": wall - self.clock.now(), "current": True})
        g = getattr(rt, "procedures", None)
        procedures = ({"nodes": g.nodes, "edges": g.edges, "rules": g.rules, "tasks": g.tasks,
                       "learned_wall": g.learned_wall} if g is not None else None)
        return {"type": "memory", "scene": self.scene, "available": True, "wall": wall,
                "spatial": spatial, "landmarks": landmarks, "notes": mem.notes(),
                "observations": mem.observations()[::-1],
                "looked": sorted(mem.data["looked"]), "sessions": sessions[::-1], "procedures": procedures}

    def frame(self) -> dict[str, Any]:
        events = self._events_from(self._ev_i)
        trace = self._trace_from(self._tr_i)
        self._ev_i = len(self.log.events)
        self._tr_i = len(self._trace_rows())
        calls: list[dict[str, Any]] = []
        if self.recorder:
            calls = self.recorder.changed_since(self._call_rev)
            self._call_rev = self.recorder.rev
        self._track(trace)
        return {"type": "frame", "events": events, "trace": trace, "calls": calls, **self.state(),
                "robot_map": self.robot_map.take_new()}

    def _track(self, trace: list[dict[str, Any]]) -> None:
        """Feed the robot's own map: its pose (odometry), whether its head is scanning,
        and where it stood when it verified where something is."""
        x, z, yaw, _ = self.world.agent_pose()
        looking = any(g.skill == "look" for g in self.robot.active_goals())
        self.robot_map.pose(self.clock.now(), x, z, yaw, looking)
        for r in trace:
            place = str(r.get("place") or "")
            if r.get("type") == "place_learned" and place and place != "UNKNOWN" and not place.startswith("hand"):
                self.robot_map.found(r["t"], r["object"], place, x, z)

    def state(self) -> dict[str, Any]:
        x, z, yaw, horizon = self.world.agent_pose()
        r = self.robot
        truth = {
            "robot": {"x": x, "z": z, "yaw": yaw, "horizon": horizon, "at": None if r.moving else r.at,
                      "moving": r.moving, "between": r.between},
            "arms": {a: {"holding": r.hand[a], "phase": r.arm_phase[a]} for a in r.hand},
            "objects": self.world.objects(),
            "goals": [{"id": g.id, "skill": g.skill, "args": g.args, "status": g.status}
                      for g in r.active_goals()],
        }
        out: dict[str, Any] = {"t": round(self.clock.now(), 2), "truth": truth, "runtime": None,
                               "nav": {"route": r.route}}                   # the nav stack's planned path
        rt = self.runtime
        if rt is None:
            return out
        belief = getattr(rt, "belief", None)
        if hasattr(belief, "to_dict"):
            belief = belief.to_dict()
        handles = getattr(rt, "actions", {})
        cancel_req = {h.entry_id for h in handles.values() if h.cancel_requested} if isinstance(handles, dict) else set()
        history = getattr(rt, "history", [])
        thinking = bool(getattr(rt, "_thinking", False) or getattr(rt, "_interpreting", False)
                        or getattr(rt, "_working", False))
        speech = getattr(rt, "speech", None)
        speech_state = None
        if speech is not None and hasattr(speech, "items"):
            cur = speech.current[0] if speech.current else None
            speech_state = {"current": {"text": cur.text, "version": cur.created_for} if cur else None,
                            "queued": [{"text": i.text, "version": i.created_for} for i in speech.items]}
        stats = None
        if callable(getattr(self.planner, "stats", None)):
            try:
                stats = self.planner.stats()
            except Exception:
                stats = None
        crashed = None
        if self.task is not None and self.task.done() and not self.task.cancelled() and self.task.exception():
            crashed = "".join(traceback.format_exception(self.task.exception()))
        out["runtime"] = {
            "belief": belief, "version": self._version(),
            "control_epoch": getattr(getattr(rt, "task", None), "control_epoch", 0),
            "paused": getattr(getattr(rt, "task", None), "paused", False), "thinking": thinking,
            "reconciling": bool(getattr(rt, "_reconciling", lambda: False)()),
            "active": [self._entry(e, e.id in cancel_req) for e in history if not e.finished],
            "recent": [self._entry(e, False) for e in history[-12:] if e.tool != "say"],
            "speech": speech_state, "stats": stats, "crashed": crashed,
        }
        persona = getattr(rt, "persona", None)
        if persona is not None:
            out["runtime"]["persona"] = persona.snapshot(rt.clock.now())
        if callable(getattr(rt, "stepping", None)):
            out["runtime"]["step"] = {"on": rt.step_mode, "waiting": rt.stepping()}
        if callable(getattr(rt, "task_steps", None)):
            out["runtime"]["procedure"] = {"steps": rt.task_steps()[-6:], "guidance": rt._guidance()}
        fused = getattr(rt, "state", None)
        if fused is not None and callable(getattr(fused, "snapshot", None)):
            out["runtime"]["fused"] = fused.snapshot(
                active=[self._entry(e, e.id in cancel_req) for e in history if not e.finished],
                recent_actions=[self._entry(e, False) for e in history[-12:]],
            )
        return out

    @staticmethod
    def _entry(e: Any, cancel_requested: bool) -> dict[str, Any]:
        data = {k: v for k, v in (e.data or {}).items() if k not in ("surfaces",)}
        return {"id": e.id, "tool": e.tool, "args": e.args, "v": e.created_for, "status": e.status,
                "source": e.source, "t": round(e.t_start, 2), "late": bool(data.get("late")),
                "cancel_requested": cancel_requested, "data": data}


class Hub:
    def __init__(self, scene: str, soul: str = "robot", speed: float = 1.0, world: World | None = None) -> None:
        self.clients: set[Any] = set()
        self.world = world if world is not None else _import(WORLD)()
        self.session: Session | None = None
        self.default = scene
        self.soul = soul                             # the robot's soul, kept across sessions
        self.custom_soul: str | None = None          # the custom soul's text (Soul → Custom…)
        self.speed = speed                           # sim seconds per wall second, kept across sessions
        self.idle = False                            # True after Stop: no session until Start
        self._lock = asyncio.Lock()
        self._sent = {"head": (-1, 0.0), "top": (-1, 0.0)}
        self.persona_level = "off"                   # own goals only when you turn them on
        self.step_mode = False                       # the page's step mode, kept across sessions like persona
        self._memory_due, self._memory_sent = False, 0.0
        self.system1: Any = None
        self.s1_status, self.s1_detail = ("off", "no System 1 configured (SYSTEM1 is not set)")
        self._s1_event = False                       # an arrival, a look or a scene change: observe now
        self.s1_gate = FrameGate()                   # which head-camera frames System 1 gets
        self._s1_sent: list[dict[str, Any]] = []     # frames sent since the last observe: the page shows them

    def start_system1(self) -> None:
        """Load the System 1 plug-in, if one is configured, and keep it running."""
        if not SYSTEM1:
            return
        try:
            self.system1 = _import(SYSTEM1)(self._on_s1_status)
        except Exception as e:
            self._on_s1_status("error", f"could not load {SYSTEM1}: {e}")
            return
        asyncio.create_task(self.system1.run())
        asyncio.create_task(self._system1_feed())

    def _on_s1_status(self, status: str, detail: str = "") -> None:
        self.s1_status, self.s1_detail = status, detail
        broadcast(self.clients, dumps({"type": "system1", "status": status, "detail": detail}))

    def _s1_ready(self) -> bool:
        return self.system1 is not None and getattr(self.system1, "status", "") == "ready"

    async def _system1_feed(self) -> None:
        """Keep System 1 current: the fused state when it changes, a head-camera frame only
        when the frame gate says it shows something new (brains/frame_gate.py: a changed
        scene or a new view, at most one a second), and a request for observations every
        few seconds while there are new frames, or at once after a scene change, an
        arrival or a look."""
        last_ctx, last_ctx_t, last_rev, last_obs_t, last_own_t = None, 0.0, -1, 0.0, -1e9
        new_frames, observing = 0, None
        while True:
            await asyncio.sleep(0.25)
            s, s1 = self.session, self.system1
            if not self._s1_ready() or s is None or s.runtime is None:
                continue
            now = time.monotonic()
            try:
                ctx = s.runtime.system1_context()
                if ctx != last_ctx and now - last_ctx_t >= 1.0:
                    await s1.update(ctx)
                    last_ctx, last_ctx_t = ctx, now
                rev = self.world.frame_rev
                if rev != last_rev:
                    frame, pose = await self.world.call(lambda: (self.world.frame(), self.world.agent_pose()))
                    if any(t in ("pick", "place") for t in ctx.get("running") or []):
                        last_own_t = now
                    own = now - last_own_t < S1_OWN_GRACE_S     # and the check right after it
                    d = self.s1_gate.decide(frame, pose, now, expected=own)
                    if d.reason != "too soon":                 # too soon: look at this frame again next time
                        last_rev = rev
                    if d.send:
                        jpeg = await self.world.call(self.world.jpeg, "head")
                        if jpeg:
                            # belief keeps the last keypoint while driving; a frame taken on the way
                            # isn't "at" it, or observations get filed where the robot started
                            moving = ctx.get("moving") or ctx.get("between")
                            where = None if moving else ctx.get("at")
                            await s1.frame(jpeg, where)
                            new_frames += 1
                            self._s1_sent.append({"url": keep_image(jpeg), "at": where, "why": d.reason,
                                                  "t": round(s.clock.now(), 1)})
                            if d.reason == "scene changed":    # something changed in front of it: ask now
                                self._s1_event = True
                due = now - last_obs_t >= S1_OBSERVE_EVERY_S or self._s1_event
                if new_frames and due and (observing is None or observing.done()):
                    self._s1_event, last_obs_t, n = False, now, new_frames
                    new_frames = 0
                    shown, self._s1_sent = self._s1_sent, []
                    observing = asyncio.create_task(self._observe(s, n, shown))
            except Exception:
                traceback.print_exc()
                await asyncio.sleep(2.0)

    async def _observe(self, s: "Session", frames: int, shown: list[dict[str, Any]] | None = None) -> None:
        t0 = time.monotonic()
        try:
            items = await self.system1.observe()
        except Exception:
            return
        if s is self.session and s.runtime is not None:
            s.runtime.add_observations(list(items or []), source="system1")
            describe = getattr(self.system1, "describe", None)
            session = f"; {describe()}" if callable(describe) else ""
            s.record_system1("observe", f"OBSERVE ({frames} new frame{'s' if frames != 1 else ''} since the last one; "
                                        f"frames so far: {self.s1_gate.summary()}{session})",
                             {"items": list(items or [])}, round(time.monotonic() - t0, 2), images=shown)

    async def _route(self, s: "Session", text: str) -> dict[str, Any] | None:
        """System 1's label for a message, as the directive the runtime reads, or None."""
        if not self._s1_ready():
            return None
        t0 = time.monotonic()
        try:
            r = await asyncio.wait_for(self.system1.route(text), S1_ROUTE_TIMEOUT_S)
        except Exception:
            r = None
        latency = round(time.monotonic() - t0, 2)
        s.record_system1("route", f"USER: {text}", r, latency)
        if not r or not r.get("kind"):
            return None
        reply = {True: "yes", False: "no"}.get(r.get("says_yes")) if isinstance(r.get("says_yes"), bool) else None
        return {"text": text, "kind": r["kind"], "source": "system1", "confidence": r.get("confidence", 0.0),
                "target": {"object": r["target"]} if r.get("target") else {}, "reply": reply,
                "replaces_task": bool(r.get("replaces_task")), "latency_s": latency}

    def meta(self) -> dict[str, Any]:
        models = [m for m in MODELS if not m[0].startswith("claude") or os.environ.get("ANTHROPIC_API_KEY")]
        return {"scenes": self.world.scenes(), "agents": AGENTS, "models": models, "souls": SOULS, "custom_soul": self.custom_soul,
                "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
                "gemini_key": bool(os.environ.get("GEMINI_API_KEY")),
                "system1": {"status": self.s1_status, "detail": self.s1_detail}}

    # ------------------------------------------------------------------
    def _notice(self, text: str) -> None:
        broadcast(self.clients, dumps({"type": "notice", "text": text}))

    # ------------------------------------------------------------------
    async def reset(self, scene: str, agent: str, model: str, forget: bool = False) -> None:
        async with self._lock:
            old, self.session = self.session, None
            if old:
                await old.stop()              # the old runtime saves its memory as it stops...
            if forget:                        # ...so forgetting has to come after that
                (ROOT / "runs" / "memory" / f"{scene}.json").unlink(missing_ok=True)
                self._notice(f"Forgot everything about {scene}: no seen objects, landmarks, looks or notes.")
            broadcast(self.clients, dumps({"type": "loading", "scene": scene}))
            self.idle = False
            new = Session(self.world, scene, agent, model, self.soul, self.speed, self.custom_soul)
            try:
                await new.start()
            except Exception:
                traceback.print_exc()
                self._notice("Could not load this room:\n" + traceback.format_exc()[-800:])
                return
            self.session = new
            self.s1_gate.reset()                      # a new room: the first frame is new again
            self._s1_sent = []
            if callable(getattr(new.runtime, "set_persona", None)):
                new.runtime.set_persona(self.persona_level)
            if callable(getattr(new.runtime, "set_step_mode", None)):
                new.runtime.set_step_mode(self.step_mode)
            self._sent = {"head": (-1, 0.0), "top": (-1, 0.0)}
            broadcast(self.clients, dumps({**new.init_message(), "meta": self.meta()}))
            broadcast(self.clients, dumps(new.memory_message()))

    async def stop_session(self) -> None:
        """End the session: the robot stops and no more model calls are made; the simulator stays warm."""
        async with self._lock:
            old, self.session = self.session, None
            if old is not None:
                if getattr(old, "clock", None) is not None:
                    old.clock.resume()                # a paused session can't wind down otherwise
                await old.stop()                      # the runtime saves its memory as it stops
            self.idle = True
        broadcast(self.clients, dumps(self.idle_message(old.scene if old is not None else self.default)))

    def idle_message(self, scene: str) -> dict[str, Any]:
        return {"type": "idle", "scene": scene, "meta": self.meta(), "soul": self.soul, "speed": self.speed}

    async def ticker(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            s = self.session
            if s is None or not self.clients:
                continue
            try:
                frame = s.frame()
                frame["system1"] = {"status": self.s1_status, "detail": self.s1_detail}
                broadcast(self.clients, dumps(frame))
                if self._s1_ready():
                    for e in frame["events"]:
                        if e.get("type") == "speech_started":
                            asyncio.create_task(self.system1.robot_said(e.get("text", "")))
                    if any(r.get("type") == "result" and r.get("skill") in ("navigate", "look")
                           and r.get("status") == "SUCCEEDED" for r in frame["trace"]):
                        self._s1_event = True
                if any(r.get("type") in MEMORY_ROWS for r in frame["trace"]):
                    self._memory_due = True
                if self._memory_due and time.monotonic() - self._memory_sent > 1.0:
                    self._memory_due, self._memory_sent = False, time.monotonic()
                    broadcast(self.clients, dumps(s.memory_message()))
                await self._send_cameras()
            except Exception:
                traceback.print_exc()

    async def _send_cameras(self, force: bool = False) -> None:
        now = time.monotonic()
        rev = self.world.frame_rev
        for which, every in (("head", HEAD_EVERY_S), ("top", TOP_EVERY_S)):
            last_rev, last_t = self._sent[which]
            if not force and (rev == last_rev or now - last_t < every):
                continue
            jpeg = await self.world.call(self.world.jpeg, which)
            if not jpeg:
                continue
            self._sent[which] = (rev, now)
            broadcast(self.clients, dumps({"type": "camera", "which": which,
                                           "jpeg": base64.b64encode(jpeg).decode()}))

    # ------------------------------------------------------------------
    async def handler(self, ws: Any) -> None:
        self.clients.add(ws)
        try:
            if self.session is not None:
                await ws.send(dumps({**self.session.init_message(), "meta": self.meta()}))
                await ws.send(dumps(self.session.memory_message()))
                await self._send_cameras(force=True)
            elif self.idle:
                await ws.send(dumps(self.idle_message(self.default)))
            else:
                await ws.send(dumps({"type": "loading", "scene": self.default, "meta": self.meta()}))
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                await self._on_message(ws, msg)
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)

    async def _on_message(self, ws: Any, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        s = self.session
        try:
            if kind == "say" and s is not None:
                text = str(msg.get("text", "")).strip()
                if not text:
                    return
                if is_stop(text) and s.runtime is not None:
                    s.runtime.emergency_stop(text, source="keyword")   # motors first; System 1 labels it next
                s.hear(text, await self._route(s, text))            # every message goes through System 1
            elif kind == "hello" and s is not None:          # the page missed the first init
                await ws.send(dumps({**s.init_message(), "meta": self.meta()}))
                await self._send_cameras(force=True)
            elif kind == "persona":                            # off, quiet, medium, optimize
                level = str(msg.get("level") or "medium")
                if level not in ("off", "quiet", "medium", "optimize"):
                    raise ValueError(f"unknown persona level {level!r}")
                self.persona_level = level
                if s is not None and callable(getattr(s.runtime, "set_persona", None)):
                    s.runtime.set_persona(level)
            elif kind == "soul":                              # the robot's soul: one of SOULS
                name = str(msg.get("name") or "")
                if name not in dict(SOULS):
                    raise ValueError(f"unknown soul {name!r}")
                make_soul(name, self.custom_soul)            # fails before anything changes
                self.soul = name
                if s is not None:
                    s.set_soul(name, self.persona_level, self.custom_soul)
                broadcast(self.clients, dumps({"type": "soul", "name": name}))
            elif kind == "soul_write":                        # Custom…: the LLM writes a soul from a description
                if s is not None and s.soul_client is not None:
                    client = s.soul_client
                else:
                    client = client_from_options({"model": MODELS[0][0], "temperature": "none", "thinking_budget": "0"},
                                                 REFERENCE_DEFAULT_MODEL)
                text = await write_soul(client, str(msg.get("description") or ""))
                await ws.send(dumps({"type": "soul_draft", "text": text}))
            elif kind == "soul_custom":                       # Custom…: use this text as the robot's soul
                text = str(msg.get("text") or "").strip()
                if not text or len(text) > CUSTOM_SOUL_MAX:
                    raise ValueError(f"a custom soul needs some text, at most {CUSTOM_SOUL_MAX} characters")
                if not parse_soul("robot_custom", text).prose:
                    raise ValueError("the custom soul has no description of who the robot is")
                self.custom_soul, self.soul = text + "\n", "custom"
                if s is not None:
                    s.set_soul("custom", self.persona_level, self.custom_soul)
                broadcast(self.clients, dumps({"type": "soul", "name": "custom", "text": self.custom_soul}))
            elif kind == "speed":                             # 1, 0.5, 0.25: slow it down to talk over it
                speed = float(msg.get("value") or 1.0)
                if not 0.05 <= speed <= 4.0:
                    raise ValueError(f"speed {speed} is out of range (0.05 to 4)")
                self.speed = speed
                if s is not None:
                    s.clock.set_speed(speed)
                broadcast(self.clients, dumps({"type": "speed", "value": speed}))
            elif kind in ("pause", "resume") and s is not None:   # sim time stands still, or goes on
                s.clock.pause() if kind == "pause" else s.clock.resume()
                broadcast(self.clients, dumps({"type": "paused", "paused": s.clock.paused}))
            elif kind == "stop_session":                      # end it: the robot stops, no more model calls
                await self.stop_session()
            elif kind == "step":                              # on, off, next
                mode = str(msg.get("mode") or "next")
                if mode in ("on", "off"):
                    self.step_mode = mode == "on"
                if s is not None and s.runtime is not None and callable(getattr(s.runtime, "step", None)):
                    if mode == "next":
                        s.runtime.step()
                    else:
                        s.runtime.set_step_mode(self.step_mode)
            elif kind == "memory" and s is not None:
                await ws.send(dumps(s.memory_message()))
            elif kind == "relearn" and s is not None and getattr(s.runtime, "procedures", None) is not None:
                g = s.runtime.procedures                     # recount from every episode on disk; rules are kept
                g.learn(await asyncio.to_thread(load_episodes))
                g.save()
                broadcast(self.clients, dumps(s.memory_message()))
            elif kind == "reset":
                await self.reset(msg.get("scene") or self.default, msg.get("agent") or "agent",
                                 msg.get("model") or MODELS[0][0], forget=bool(msg.get("forget")))
        except Exception as e:
            await ws.send(dumps({"type": "notice", "text": str(e)}))


def process_request(connection: Any, request: Any) -> Any:
    path = request.path.split("?")[0]
    if path == "/ws":
        return None
    if path in ("/", "/index.html"):
        resp = connection.respond(http.HTTPStatus.OK, INDEX.read_text())
        del resp.headers["Content-Type"]
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp
    m = re.fullmatch(r"/s1/(\d+)\.jpg", path)                    # a frame System 1 was shown
    if m:
        jpeg = S1_IMAGES.get(m.group(1))
        if jpeg is None:
            return connection.respond(http.HTTPStatus.NOT_FOUND, "gone (only the newest frames are kept)\n")
        return Response(200, "OK", Headers([("Content-Type", "image/jpeg"), ("Content-Length", str(len(jpeg))),
                                            ("Cache-Control", "max-age=3600")]), jpeg)
    m = re.fullmatch(r"/views/([a-z_]+\.js)", path)            # the page's view scripts, nothing else
    if m and (VIEWS / m.group(1)).is_file():
        resp = connection.respond(http.HTTPStatus.OK, (VIEWS / m.group(1)).read_text())
        del resp.headers["Content-Type"]
        resp.headers["Content-Type"] = "text/javascript; charset=utf-8"
        resp.headers["Cache-Control"] = "no-cache"
        return resp
    return connection.respond(http.HTTPStatus.NOT_FOUND, "not found\n")


async def main() -> None:
    global PERCEPTION
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--scene", help="a room the world offers; the first one if not given. For AI2-THOR, an iTHOR "
                                     "room (FloorPlan1 kitchen, FloorPlan301 bedroom) or a ProcTHOR house (procthor-train-7)")
    ap.add_argument("--soul", default="robot", choices=[n for n, _ in SOULS if n != "custom"],
                    help="the robot's soul, personas/<soul>.md (the page's Soul menu changes it too)")
    ap.add_argument("--soul-file", type=Path,
                    help="a custom soul in the personas/*.md format; it becomes the Soul menu's Custom… entry and is used")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="sim seconds per wall second: 0.5 runs the robot at half speed (the models still answer in wall time)")
    ap.add_argument("--world", default=WORLD,
                    help="the world, as module:factory (sim/world.py); the default is AI2-THOR")
    ap.add_argument("--perception", default=PERCEPTION,
                    help="how the robot sees, as module:factory taking the world (perception/source.py); "
                         "the default is the stand-in detector")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    PERCEPTION = args.perception
    world = _import(args.world)()
    scene = args.scene or world.scenes()[0][0]
    hub = Hub(scene, args.soul, args.speed, world=world)
    if args.soul_file:
        hub.custom_soul, hub.soul = args.soul_file.read_text(), "custom"
    atexit.register(hub.world.close)             # last resort; close() is safe to call twice
    try:
        try:
            server = await serve(hub.handler, args.host, args.port, process_request=process_request, max_size=2 ** 22)
        except OSError as e:
            print(f"Port {args.port} is in use ({e.strerror}): a playground server is probably already running "
                  f"at http://{args.host}:{args.port}. Stop it first; no simulator was started.", file=sys.stderr)
            return
        hub.start_system1()
        async with server:
            print(f"Worldline on http://{args.host}:{args.port}  (loading {scene}…)", flush=True)
            await hub.reset(scene, "agent", MODELS[0][0])
            if hub.session and hub.session.error:
                print(hub.session.error, file=sys.stderr)
            print("ready", flush=True)
            await hub.ticker()
    finally:
        if hub.session is not None:
            try:
                await asyncio.wait_for(hub.session.stop(), 5)   # the runtime saves memory as it stops
            except Exception:
                pass
        hub.world.close()                        # stop and reap the simulator
        print("stopped", flush=True)


def _on_signal(signum: int, frame: Any) -> None:
    # `kill` or a closed terminal should shut down like Ctrl-C, so main()'s finally
    # stops the Unity process; otherwise every restart leaves an orphaned simulator.
    raise KeyboardInterrupt


if __name__ == "__main__":
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    # main() has saved memory and stopped the simulator. A simulator thread can still be
    # blocked inside ai2thor (a load that never connected); don't wait for it to exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
