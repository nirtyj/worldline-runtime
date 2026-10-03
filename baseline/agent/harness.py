"""Baseline harness: a sequential loop around the brain.

  1. Wait for the user to say something.
  2. Ask the brain for the next tool call and run it until it finishes.
  3. Repeat until the brain says "wait"; then go back to 1.

Anything the user says while a step is running is read before the next step.
The robot's built-in pick and place goals do the manipulation.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

from brains.interface import BrainInput, HistoryEntry, ToolCall
from sim.goals import GoalRejected

MAX_STEPS = 60


class BaselineRuntime:
    def __init__(self, robot: Any, user: Any, brain: Any, clock: Any) -> None:
        self.robot, self.user, self.brain, self.clock = robot, user, brain, clock
        self.map = robot.lookup_keypoints()
        self.utterances: list[Any] = []
        self.kinds: dict[str, str] = {}
        self.version = 0
        self.history: list[HistoryEntry] = []
        self._ids = itertools.count(1)
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._working = False
        self._log: list[dict[str, Any]] = []
        base = robot.base_state()
        self.belief: dict[str, Any] = {
            "robot": {"at": base["at"], "between": None},
            "hands": {arm: {"holding": None, "verified": True, "source": "start", "t": 0.0}
                      for arm in ("left", "right")},
            "objects": {},
            "blocked": [],
        }
        for m in robot.memory():
            self.belief["objects"][m["id"]] = {
                "type": m.get("type"), "brand": m.get("brand"), "color": m.get("color"),
                "label": " ".join(p for p in (m.get("color"), m.get("brand"), m.get("type")) if p),
                "where": m.get("surface"), "x": None, "depth": None, "seen_from": None,
                "verified": False, "source": "memory", "t": m.get("last_seen_t", 0.0)}

    # ------------------------------------------------------------------
    async def run(self) -> None:
        listener = asyncio.create_task(self._listen())
        try:
            while True:
                utt = await self._inbox.get()
                self._working = True
                try:
                    await self._hear(utt)
                    await self._work()
                finally:
                    self._working = False
        finally:
            listener.cancel()

    def idle(self) -> bool:
        return not self._working and self._inbox.empty()

    def trace(self) -> list[dict[str, Any]]:
        return self._log

    async def _listen(self) -> None:
        while True:
            self._inbox.put_nowait(await self.user.next())

    async def _hear(self, utt: Any) -> None:
        self.utterances.append(utt)
        kind = await self.brain.classify(utt, self._ctx())
        self.kinds[utt.id] = kind
        if kind in ("request", "correction"):
            self.version += 1
        self._log.append({"t": self.clock.now(), "type": "heard", "id": utt.id, "kind": kind})

    async def _work(self) -> None:
        for _ in range(MAX_STEPS):
            while not self._inbox.empty():
                await self._hear(self._inbox.get_nowait())
            call = await self.brain.next_action(self._ctx())
            self._log.append({"t": self.clock.now(), "type": "decision", "tool": call.tool, "args": call.args})
            if call.tool == "wait":
                if self._inbox.empty():
                    return
                continue
            await self._execute(call)

    def _ctx(self) -> BrainInput:
        return BrainInput(now=self.clock.now(), map=self.map, utterances=list(self.utterances),
                          kinds=dict(self.kinds), intent_version=self.version, belief=self.belief,
                          history=list(self.history), active=[], paused=False, note=None)

    async def _execute(self, call: ToolCall) -> None:
        entry = HistoryEntry(id=next(self._ids), tool=call.tool, args=dict(call.args),
                             created_for=self.version, t_start=self.clock.now(), status="running",
                             tag=call.tag)
        self.history.append(entry)
        try:
            goal = self.robot.send_goal(call.tool, **call.args)
        except GoalRejected as e:
            entry.status, entry.data, entry.t_end = "REJECTED", {"reason": e.reason}, self.clock.now()
            return
        except TypeError as e:
            entry.status, entry.data, entry.t_end = "REJECTED", {"reason": str(e)}, self.clock.now()
            return
        result = await goal.result()
        entry.status, entry.data, entry.t_end = result.status, dict(result.data), self.clock.now()
        if result.status == "SUCCEEDED":
            self._update(call, result.data)

    def _update(self, call: ToolCall, data: dict[str, Any]) -> None:
        b = self.belief
        now = self.clock.now()
        if call.tool == "navigate":
            b["robot"] = {"at": data.get("at"), "between": data.get("between")}
        elif call.tool == "look":
            for surface, items in (data.get("surfaces") or {}).items():
                for v in items:
                    b["objects"][v["id"]] = {"type": v["type"], "brand": v["brand"], "color": v["color"],
                                             "label": v["label"], "where": surface, "x": v["x"],
                                             "depth": v["depth"], "seen_from": data.get("at"),
                                             "verified": True, "source": "look", "t": now}
            for arm, oid in (data.get("hands") or {}).items():
                b["hands"][arm] = {"holding": oid, "verified": True, "source": "look", "t": now}
        elif call.tool == "pick":
            arm, oid = call.args["arm"], call.args["object"]
            b["hands"][arm] = {"holding": oid, "verified": True, "source": "pick", "t": now}
            if oid in b["objects"]:
                b["objects"][oid].update(where=f"hand:{arm}", source="pick", t=now)
        elif call.tool == "place":
            arm, oid = call.args["arm"], call.args["object"]
            b["hands"][arm] = {"holding": None, "verified": True, "source": "place", "t": now}
            at = b["robot"]["at"]
            surface = next((s for s, i in self.map["surfaces"].items()
                            if at in i["keypoints"] and i["height_m"] <= self.map["max_reach_height_m"]), None)
            if oid in b["objects"]:
                b["objects"][oid].update(where=surface, source="place", t=now)
