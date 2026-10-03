"""The robot API the runtime drives, backed by an AI2-THOR room.

The contract (goals from sim/goals.py) that agent/ drives:

    goal = robot.send_goal("navigate", to="counter_2a")   # may raise GoalRejected
    goal.cancel()                                          # a REQUEST
    result = await goal.result()                           # SUCCEEDED / ABORTED / CANCELED

What "cancel" means per skill:
  navigate      stops at the next grid point (every 0.25 m), possibly between keypoints
  look, reachability   stop immediately (a look scans: three headings, two tilts)
  pick / place  finish the chunk that is running (about 1 s), then stop. A cancel
                that lands during "close" still ends with the object in the hand:
                the result is CANCELED with holding=True.
  say           cuts off immediately

``halt()`` stops the base and the arm at once; running motion ends ABORTED.

The robot has two arm names, but AI2-THOR's agent has one hand, so it can hold
one object at a time: reachability answers "right" when the hand is free.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import traceback
from typing import Any

from sim.goals import Goal, GoalRejected, GoalResult

from .world import GRID, MAX_REACH_HEIGHT, NAV_SPEED, REACH_M, ThorWorld

ARMS = ("left", "right")
SKILLS = ("navigate", "look", "reachability", "pick", "place", "say")
PICK_CHUNKS = (("approach", 1.0), ("pregrasp", 0.8), ("close", 0.9), ("lift", 0.9))
PLACE_CHUNKS = (("lower", 1.0), ("open", 0.8), ("retract", 0.9))
SCAN_YAWS = (0.0, -40.0, 40.0)      # a look turns the head to each of these...
SCAN_TILT = 25.0                    # ...at its usual tilt and this much lower
SCAN_STEP_S = 0.05            # each view also costs a render (~130 ms)
FOV_DEG = 90.0


class ThorRobot:
    def __init__(self, world: ThorWorld, clock: Any, log: Any) -> None:
        self.world = world
        self.clock = clock
        self.log = log
        self._goal_ids = itertools.count(1)
        self._subs: list[asyncio.Queue] = []
        self._active: set[Goal] = set()
        self._tasks: set[asyncio.Task] = set()
        self._base_goal: Goal | None = None
        self._arm_goal: dict[str, Goal | None] = {a: None for a in ARMS}
        self._playing: list[Goal] = []
        self._halt_epoch = 0
        self._map: tuple[Any, dict[str, Any]] | None = None
        self._scanned: tuple[str | None, set[str]] = (None, set())   # the last look: where, and what it saw
        self.at: str | None = "start"
        self.between: list[str] | None = None
        self.moving = False
        self.route: list[list[float]] | None = None   # the nav stack's planned path while driving
        self.hand: dict[str, str | None] = {a: None for a in ARMS}   # which arm name holds the object
        self.arm_phase: dict[str, str] = {a: "home" for a in ARMS}

    # ------------------------------------------------------------------
    # Static knowledge
    # ------------------------------------------------------------------
    def lookup_keypoints(self) -> dict[str, Any]:
        """The map: spots, the surfaces reached from them, rooms, people, and the
        path length between every pair of spots. Fixed per room load, so it's
        computed once (perception asks for it ten times a second)."""
        lay = self.world.layout
        if self._map is not None and self._map[0] is lay:
            return self._map[1]
        from .procthor import room_at
        kps = lay.keypoints()
        keypoints = {"start": {"desc": "where the robot started, next to you", "xy": [kps["start"][0], kps["start"][1]]}}
        for s in lay.surfaces.values():
            keypoints[s.name] = {"desc": f"in front of the {s.desc}", "xy": [s.stand[0], s.stand[1]]}
        if lay.rooms:
            for k, v in keypoints.items():
                v["room"] = room_at(lay.rooms, v["xy"][0], v["xy"][1])
        names = list(keypoints)
        edges = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = self.world.path_length(tuple(keypoints[a]["xy"]), tuple(keypoints[b]["xy"]))
                if d is not None:
                    edges.append([a, b, d])
        surfaces = {s.name: {"keypoints": [s.name], "desc": f"the {s.desc}", "height_m": s.height,
                             "xy": [round(s.center[0], 2), round(s.center[1], 2)]}
                    for s in lay.surfaces.values()}
        people = {}
        if lay.user_surface:
            people["user"] = {"deliver_to_surface": lay.user_surface, "keypoint": lay.user_surface}
        rooms = {name: {"label": r["label"], "spots": [k for k, v in keypoints.items() if v.get("room") == name]}
                 for name, r in lay.rooms.items()}
        out = {"scene": lay.scene, "keypoints": keypoints, "edges": edges, "surfaces": surfaces, "people": people,
               "rooms": rooms, "nav_speed_mps": NAV_SPEED, "max_reach_height_m": MAX_REACH_HEIGHT}
        self._map = (lay, out)
        return out

    def memory(self) -> list[dict[str, Any]]:
        """The robot has no memory of the room of its own; the runtime keeps one
        from what the robot has seen (agent/memory.py). Nothing here reads the
        simulator's truth, so the room starts unknown."""
        return []

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------
    def base_state(self) -> dict[str, Any]:
        x, z, _, _ = self.world.agent_pose()
        return {"moving": self.moving, "at": None if self.moving else self.at,
                "between": list(self.between) if self.between else None,
                "xy": [round(x, 3), round(z, 3)]}

    def gripper(self, arm: str) -> dict[str, Any]:
        if self.hand[arm] is None:
            return {"closed": False, "width": 0.085, "force": 0.0}
        return {"closed": True, "width": 0.06, "force": 8.0}      # a gripper feels, it can't name

    def proprio(self, arm: str) -> dict[str, Any]:
        return {"joints": [0.0] * 7, "gripper": "closed" if self.hand[arm] else "open",
                "posture": self.arm_phase[arm]}

    def telemetry(self) -> dict[str, Any]:
        """Latest proprioceptive state, shaped like a real robot adapter."""
        x, z, yaw, horizon = self.world.agent_pose()
        active = sorted(({"id": g.id, "skill": g.skill, "status": g.status}
                         for g in self.active_goals()), key=lambda g: g["id"])
        return {
            "pose": {"x": round(x, 3), "z": round(z, 3), "yaw": round(yaw, 2),
                     "horizon": round(horizon, 2), "at": None if self.moving else self.at,
                     "between": list(self.between) if self.between else None},
            "velocity": {"linear_mps": NAV_SPEED if self.moving else 0.0, "angular_dps": 0.0},
            "moving": self.moving,
            "arms": {arm: self.proprio(arm) for arm in ARMS},
            "grippers": {arm: self.gripper(arm) for arm in ARMS},
            "active_skills": active,
            "health": {"ok": True, "source": "thor"},
        }

    def perception(self) -> dict[str, Any]:
        """What the head camera sees right now, as a perception stack would report
        it: only objects and landmarks in view, with exact labels and positions."""
        rx, rz, _, _ = self.world.agent_pose()
        objects: dict[str, dict[str, Any]] = {}
        for oid, raw in self.world.objects().items():
            if not raw.get("visible"):
                continue
            x, y, z = float(raw["x"]), float(raw["y"]), float(raw["z"])
            objects[oid] = {
                "type": raw["type"], "label": raw["label"], "where": raw["where"],
                "pose": {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)},
                "distance_m": round(math.dist((rx, rz), (x, z)), 3),
                "visible": bool(raw.get("visible", False)), "confidence": 1.0,
                "source": "thor-ground-truth",
            }
        people: dict[str, dict[str, Any]] = {}
        for name, info in self.lookup_keypoints().get("people", {}).items():
            point = self.world.layout.keypoints().get(info["keypoint"])
            if point:
                people[name] = {
                    "pose": {"x": round(point[0], 3), "z": round(point[1], 3)},
                    "distance_m": round(math.dist((rx, rz), (point[0], point[1])), 3),
                    "confidence": 1.0, "source": "thor-ground-truth",
                    **info,
                }
        landmarks: dict[str, dict[str, Any]] = {}
        for name, lm in self.world.landmarks().items():
            if not lm["visible"]:
                continue
            landmarks[name] = {
                "type": lm["type"], "label": lm["label"], "near": lm["near"],
                "pose": {"x": round(lm["x"], 3), "y": round(lm["y"], 3), "z": round(lm["z"], 3)},
                "distance_m": round(math.dist((rx, rz), (lm["x"], lm["z"])), 3),
                "visible": lm["visible"], "confidence": 1.0, "source": "thor-ground-truth",
            }
        return {"objects": objects, "landmarks": landmarks, "people": people, "source": "thor-ground-truth"}

    def events(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subs.append(queue)
        return queue

    def active_goals(self) -> list[Goal]:
        return sorted(self._active, key=lambda g: g.id)

    def idle(self) -> bool:
        return not self._active

    # ------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------
    def send_goal(self, skill: str, **args: Any) -> Goal:
        try:
            self._validate(skill, args)
        except GoalRejected as e:
            self.log.emit("goal_rejected", skill=skill, args=args, reason=e.reason)
            raise
        goal = Goal(self, skill, args)
        self._active.add(goal)
        if skill == "navigate":
            self._base_goal = goal
        elif skill in ("pick", "place"):
            self._arm_goal[args["arm"]] = goal
        self._public("goal_accepted", goal=goal.id, skill=skill, args=dict(args), **self._obj(args.get("object")))
        coro = {
            "navigate": lambda: self._navigate(goal, args["to"]),
            "look": lambda: self._look(goal, bool(args.get("glance"))),
            "reachability": lambda: self._reachability(goal, args["object"]),
            "pick": lambda: self._pick(goal, args["object"], args["arm"]),
            "place": lambda: self._place(goal, args["object"], args["arm"]),
            "say": lambda: self._say(goal, args["text"]),
        }[skill]()
        self._start(goal, coro)
        return goal

    def halt(self) -> dict[str, Any]:
        """Hardware stop: base and arm freeze now. Motion goals end ABORTED."""
        self._halt_epoch += 1
        self._public("halted")
        return {"accepted": True, "stopped": True, "source": "thor"}

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    def _validate(self, skill: str, args: dict[str, Any]) -> None:
        if skill not in SKILLS:
            raise GoalRejected(f"unknown skill {skill!r}; skills: {', '.join(SKILLS)}")
        required = {"navigate": ["to"], "reachability": ["object"], "pick": ["object", "arm"],
                    "place": ["object", "arm"], "say": ["text"], "look": []}[skill]
        missing = [k for k in required if k not in args]
        if missing:
            raise GoalRejected(f"missing argument(s): {', '.join(missing)}")
        arm_busy = any(self._arm_goal[a] is not None for a in ARMS)
        if skill == "navigate":
            if args["to"] not in self.world.layout.keypoints():
                raise GoalRejected(f"unknown keypoint {args['to']!r}")
            if self._base_goal is not None:
                raise GoalRejected("base_busy")
            if arm_busy:
                raise GoalRejected("arm_busy")
        if skill in ("reachability", "pick", "place") and args["object"] not in self.world.layout.obj_ids:
            raise GoalRejected(f"unknown object {args['object']!r}")
        if skill in ("pick", "place"):
            arm = args["arm"]
            if arm not in ARMS:
                raise GoalRejected(f"unknown arm {arm!r}")
            if self.moving or self._base_goal is not None:
                raise GoalRejected("base_moving")
            if arm_busy:
                raise GoalRejected("arm_busy")
            if skill == "pick" and any(self.hand.values()):
                raise GoalRejected("hand_full: this robot holds one object at a time")
            if skill == "place" and self.hand[arm] != args["object"]:
                raise GoalRejected("nothing_in_gripper" if self.hand[arm] is None else "holding_something_else")
        if skill == "say" and not str(args["text"]).strip():
            raise GoalRejected("empty text")

    def _obj(self, short: str | None) -> dict[str, Any]:
        return {"object": short} if short else {}

    def _public(self, type: str, **fields: Any) -> None:
        event = self.log.emit(type, **fields)
        for queue in list(self._subs):
            queue.put_nowait(dict(event))

    def _start(self, goal: Goal, coro) -> None:
        async def runner() -> None:
            try:
                result = await coro
            except asyncio.CancelledError:
                self._finish(goal, GoalResult("ABORTED", {"reason": "shutdown"}))
                raise
            except Exception as e:
                self.log.emit("internal_error", goal=goal.id, error=repr(e), tb=traceback.format_exc())
                result = GoalResult("ABORTED", {"reason": f"internal_error: {e!r}"})
            self._finish(goal, result)

        task = asyncio.create_task(runner())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _finish(self, goal: Goal, result: GoalResult) -> None:
        if goal.done:
            return
        goal.status = result.status
        self._active.discard(goal)
        if self._base_goal is goal:
            self._base_goal = None
        for a in ARMS:
            if self._arm_goal[a] is goal:
                self._arm_goal[a] = None
        if goal in self._playing:
            self._playing.remove(goal)
        goal._future.set_result(result)
        self._public("goal_done", goal=goal.id, skill=goal.skill, args=dict(goal.args),
                     status=result.status, data=dict(result.data), **self._obj(goal.args.get("object")))

    async def _wait(self, goal: Goal, seconds: float, epoch: int, cancellable: bool = True) -> str | None:
        """Sleep through a skill step. Returns "halted" or "canceled" if that happened first."""
        t_end = self.clock.now() + seconds
        while self.clock.now() < t_end:
            if self._halt_epoch != epoch:
                return "halted"
            if cancellable and goal.cancel_requested:
                return "canceled"
            await self.clock.sleep(min(0.05, max(0.0, t_end - self.clock.now())))
        return "halted" if self._halt_epoch != epoch else None

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------
    async def _say(self, goal: Goal, text: str) -> GoalResult:
        duration = 0.3 + 0.32 * max(1, len(str(text).split()))
        if self._playing:
            self.log.emit("speech_overlap", goal=goal.id, with_goals=[g.id for g in self._playing], text=text)
        self._playing.append(goal)
        self._public("speech_started", goal=goal.id, text=text)
        t_end = self.clock.now() + duration
        while self.clock.now() < t_end:
            if goal.cancel_requested:
                played = 1.0 - (t_end - self.clock.now()) / duration
                self._public("speech_cut", goal=goal.id, text=text, played=round(played, 2))
                return GoalResult("CANCELED", {"played": round(played, 2)})
            await self.clock.sleep(0.05)
        self._public("speech_ended", goal=goal.id, text=text)
        return GoalResult("SUCCEEDED", {})

    async def _navigate(self, goal: Goal, to: str) -> GoalResult:
        epoch = self._halt_epoch
        kps = self.world.layout.keypoints()
        tx, tz, tyaw, thor = kps[to]
        x, z, yaw, _ = self.world.agent_pose()
        path = self.world.path((x, z), (tx, tz))
        start_name = self.at
        if path is None:
            return GoalResult("ABORTED", {"reason": "no_path", "at": self.at})
        self.moving, self.at, self.between = True, None, None
        self.route = [[round(px, 2), round(pz, 2)] for px, pz in path] + [[round(tx, 2), round(tz, 2)]]
        self._public("base_moving", to=to)
        try:
            prev = (x, z)
            for step in path[1:]:
                heading = math.degrees(math.atan2(step[0] - prev[0], step[1] - prev[1])) % 360
                t0 = self.clock.now()
                await self.world.call(self.world.teleport, step[0], step[1], heading, 15.0)
                # the render already took part of this step's time; keep the speed at NAV_SPEED
                why = await self._wait(goal, max(0.0, GRID / NAV_SPEED - (self.clock.now() - t0)), epoch)
                prev = step
                if why:
                    here = self._keypoint_at(step)
                    self.at, self.between = here, None if here else [start_name or "start", to]
                    self._public("base_stopped", at=here, between=self.between)
                    status = "ABORTED" if why == "halted" else "CANCELED"
                    return GoalResult(status, {"reason": why, "at": here, "between": self.between})
            await self.world.call(self.world.teleport, tx, tz, tyaw, thor)
        finally:
            self.moving, self.route = False, None
        self.at, self.between = to, None
        self._public("base_stopped", at=to)
        return GoalResult("SUCCEEDED", {"at": to})

    def _keypoint_at(self, xz: tuple[float, float]) -> str | None:
        for name, (x, z, _, _) in self.world.layout.keypoints().items():
            if math.dist((x, z), xz) < 0.05:
                return name
        return None

    async def _look(self, goal: Goal, glance: bool = False) -> GoalResult:
        """Scan: turn the head to three headings at two tilts and report what the
        camera saw, grouped by where each thing is, plus the views it covered, so
        the runtime can tell "looked there and it's gone" from "never looked".

        A glance looks straight ahead only, at the usual tilt and lower (a nod,
        no turning), to check a pick or a place in front of the robot. It
        reports what it saw and the hands but no views: something not in one
        straight-ahead view (behind the coffee machine, say) isn't gone."""
        epoch = self._halt_epoch
        if self.moving:
            return GoalResult("SUCCEEDED", {"at": None, "surfaces": {}, "landmarks": [], "views": [],
                                            "hands": dict(self.hand)})
        at = self.at
        x, z, yaw0, tilt0 = self.world.agent_pose()
        seen: dict[str, dict[str, Any]] = {}
        marks: dict[str, dict[str, Any]] = {}
        views = []
        tilts = [tilt0, min(tilt0 + SCAN_TILT, 60.0)]
        if at in self.world.layout.alias.values():
            tilts.append(0.0)                 # a shelf unit: look straight at its upper levels too
        yaws = SCAN_YAWS
        if glance:
            yaws, tilts = (0.0,), tilts[:2]
        try:
            for dy in yaws:
                for tilt in tilts:
                    heading = (yaw0 + dy) % 360
                    await self.world.call(self.world.teleport, x, z, heading, tilt)   # a fresh render
                    views.append({"x": round(x, 2), "z": round(z, 2), "yaw": round(heading, 1),
                                  "tilt": tilt, "fov": FOV_DEG, "range": REACH_M})
                    for short, o in self.world.objects().items():
                        if o["visible"] and o["where"] != "hand":
                            seen[short] = o
                    for name, lm in self.world.landmarks().items():
                        if lm["visible"]:
                            marks[name] = lm
                    why = await self._wait(goal, SCAN_STEP_S, epoch)
                    if why:
                        return GoalResult("CANCELED" if why == "canceled" else "ABORTED", {"reason": why})
        finally:
            await self.world.call(self.world.teleport, x, z, yaw0, tilt0)   # face forward again
        # what counts as "seen here" for reachability: a glance adds to the last scan of this spot
        self._scanned = (at, set(seen) | (self._scanned[1] if glance and self._scanned[0] == at else set()))
        surfaces: dict[str, list[dict[str, Any]]] = {}
        for short, o in sorted(seen.items()):
            surfaces.setdefault(o["where"], []).append(
                {"id": short, "type": o["type"], "brand": None, "color": None, "label": o["label"],
                 "surface": o["where"], "pos": [round(o["x"], 2), round(o["z"], 2)]})
        landmarks = [{"id": name, "type": lm["type"], "label": lm["label"], "near": lm["near"],
                      "pos": [round(lm["x"], 2), round(lm["z"], 2)]} for name, lm in sorted(marks.items())]
        return GoalResult("SUCCEEDED", {"at": at, "surfaces": surfaces, "landmarks": landmarks,
                                        "views": [] if glance else views, "hands": dict(self.hand)})

    async def _reachability(self, goal: Goal, short: str) -> GoalResult:
        if await self._wait(goal, 0.3, self._halt_epoch) == "canceled":
            return GoalResult("CANCELED", {})
        base = {"object": short, "at": self.at}
        if self.moving or self.at is None:
            return GoalResult("SUCCEEDED", {**base, "reachable": False, "reason": "base_moving"})
        o = self.world.objects().get(short)
        if o is None:
            return GoalResult("SUCCEEDED", {**base, "reachable": False, "reason": "not_found"})
        if o["where"] == "hand":
            return GoalResult("SUCCEEDED", {**base, "reachable": False, "reason": "in_hand"})
        seen_here = self._scanned[0] == self.at and short in self._scanned[1]
        if not o["visible"] and not seen_here:   # don't reveal where an unseen object is
            return GoalResult("SUCCEEDED", {**base, "reachable": False, "reason": "not_seen_here"})
        if o["where"] not in self.world.layout.surfaces:
            reason = f"inside_or_on_{o['where']}"
            return GoalResult("SUCCEEDED", {**base, "reachable": False, "reason": reason})
        if o["y"] > MAX_REACH_HEIGHT:
            return GoalResult("SUCCEEDED", {**base, "reachable": False, "reason": "too_high"})
        x, z, _, _ = self.world.agent_pose()
        if math.dist((x, z), (o["x"], o["z"])) > REACH_M:
            return GoalResult("SUCCEEDED", {**base, "reachable": False, "reason": "too_far",
                                            "suggest": o["where"]})
        if any(self.hand.values()):
            return GoalResult("SUCCEEDED", {**base, "reachable": False, "reason": "hand_full"})
        return GoalResult("SUCCEEDED", {**base, "reachable": True, "arm": "right"})

    async def _pick(self, goal: Goal, short: str, arm: str) -> GoalResult:
        epoch = self._halt_epoch
        for i, (phase, seconds) in enumerate(PICK_CHUNKS):
            if i > 0 and goal.cancel_requested:            # checked only between chunks
                return GoalResult("CANCELED", {"holding": self.hand[arm] is not None, "chunks": i})
            self.arm_phase[arm] = phase
            self.log.emit("chunk_started", arm=arm, phase=phase, object=short)
            why = await self._wait(goal, seconds, epoch, cancellable=False)   # a chunk always finishes
            if why == "halted":
                self.arm_phase[arm] = "stopped"
                return GoalResult("ABORTED", {"reason": "halted", "holding": self.hand[arm] is not None, "phase": phase})
            if phase == "close":
                ok, err = await self.world.call(self.world.pickup, short)
                if not ok:
                    self.log.emit("grasp_missed", arm=arm, object=short, reason=err[:120])
                    self.arm_phase[arm] = "home"
                    return GoalResult("ABORTED", {"reason": "grasp_failed", "detail": err[:120], "holding": False})
                self.hand[arm] = short
                self.log.emit("grasp_closed", arm=arm, object=short)
        self.arm_phase[arm] = "lifted"
        if goal.cancel_requested:
            return GoalResult("CANCELED", {"holding": True, "chunks": len(PICK_CHUNKS)})
        return GoalResult("SUCCEEDED", {"holding": True, "chunks": len(PICK_CHUNKS)})

    async def _place(self, goal: Goal, short: str, arm: str) -> GoalResult:
        epoch = self._halt_epoch
        surface = self.at if self.at in self.world.layout.surfaces else None
        if surface is None:
            return GoalResult("ABORTED", {"reason": "no_surface_here"})
        for i, (phase, seconds) in enumerate(PLACE_CHUNKS):
            if i > 0 and goal.cancel_requested:
                return GoalResult("CANCELED", {"holding": self.hand[arm] is not None, "chunks": i})
            self.arm_phase[arm] = phase
            self.log.emit("chunk_started", arm=arm, phase=phase, object=short)
            why = await self._wait(goal, seconds, epoch, cancellable=False)
            if why == "halted":
                self.arm_phase[arm] = "stopped"
                return GoalResult("ABORTED", {"reason": "halted", "holding": self.hand[arm] is not None, "phase": phase})
            if phase == "open":
                ok, err = await self.world.call(self.world.put, surface)
                if not ok:
                    self.arm_phase[arm] = "lifted"
                    return GoalResult("ABORTED", {"reason": "no_room_on_surface", "detail": err[:120], "holding": True})
                self.hand[arm] = None
                self.log.emit("object_placed", arm=arm, object=short, surface=surface)
        self.arm_phase[arm] = "home"
        return GoalResult("SUCCEEDED", {"holding": False, "surface": surface})
