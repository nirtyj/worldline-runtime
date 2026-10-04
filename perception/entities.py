"""The robot's own names for the things it has seen.

A detector says "a newspaper, here"; the runtime needs an id it can come back to ("newspaper_1").
A detection takes the id of the thing remembered close by: one of its type (or one it might be)
within SAME_THING_M, or anything within SAME_SPOT_M, since a detector may call the same folded
paper a magazine in one view and a newspaper in the next. Otherwise it gets a new id,
<type>_<n>. A thing's type is what it was called most often; its label says what else it might
be. Nothing here knows the simulator's ids: if someone carries a thing across the room while the
robot isn't looking, the robot sees "a newspaper" over there and calls it newspaper_2, as a real
robot would, until something tells it they are the same.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

SAME_THING_M = 0.45          # a detection this close to a remembered thing of its type is that thing
SAME_SPOT_M = 0.25           # ...and this close, whatever the detector called it this time


@dataclass
class Entity:
    id: str
    type: str
    label: str
    x: float
    y: float
    z: float
    where: str
    confidence: float = 1.0
    seen: int = 1
    t: float = 0.0                 # when it was last seen
    held: bool = False
    names: Counter = field(default_factory=Counter)     # what detectors called it, and how often

    def heard(self, type_: str, also: list[str] | None = None) -> None:
        self.names[type_] += 1.0
        for a in also or []:
            if a != type_:
                self.names[a] += 0.5
        top = self.names.most_common()
        self.type = top[0][0]
        others = [n for n, _ in top[1:3]]
        base = self.type.replace("_", " ")
        self.label = base + (f" (maybe {' or '.join(o.replace('_', ' ') for o in others)})" if others else "")

    def might_be(self, type_: str) -> bool:
        return type_ == self.type or type_ in self.names

    def as_report(self, visible: bool) -> dict[str, Any]:
        return {"type": self.type, "label": self.label, "where": "hand" if self.held else self.where,
                "x": self.x, "y": self.y, "z": self.z, "visible": visible, "confidence": self.confidence}


@dataclass
class Entities:
    items: dict[str, Entity] = field(default_factory=dict)

    def clear(self) -> None:
        self.items.clear()

    def _new_id(self, type_: str) -> str:
        n = 0
        for k in self.items:
            m = re.fullmatch(re.escape(type_) + r"_(\d+)", k)
            if m:
                n = max(n, int(m.group(1)))
        return f"{type_}_{n + 1}"

    def assign(self, dets: list[dict[str, Any]], t: float) -> dict[str, Entity]:
        """Give each placed detection (type, x, y, z, where, confidence, also) an id: the nearest
        remembered thing it could be (see the module's note), one detection per thing, else a new
        one. Updates what is remembered; returns {id: entity} for this view."""
        out: dict[str, Entity] = {}
        pairs = []
        for i, d in enumerate(dets):
            for e in self.items.values():
                if e.held:
                    continue
                gap = math.dist((e.x, e.z), (d["x"], d["z"]))
                same = e.might_be(d["type"]) or any(e.might_be(a) for a in d.get("also") or [])
                if gap <= (SAME_THING_M if same else SAME_SPOT_M):
                    pairs.append((0 if same else 1, gap, i, e.id))
        taken_d, taken_e = set(), set()
        for _, gap, i, eid in sorted(pairs):
            if i in taken_d or eid in taken_e:
                continue
            taken_d.add(i)
            taken_e.add(eid)
            e, d = self.items[eid], dets[i]
            e.x, e.y, e.z, e.where = d["x"], d["y"], d["z"], d["where"]
            e.confidence, e.seen, e.t = float(d.get("confidence", 1.0)), e.seen + 1, t
            e.heard(d["type"], d.get("also"))
            out[eid] = e
        for i, d in enumerate(dets):
            if i in taken_d:
                continue
            eid = self._new_id(d["type"])
            e = Entity(eid, d["type"], d["type"].replace("_", " "), d["x"], d["y"], d["z"], d["where"],
                       float(d.get("confidence", 1.0)), 1, t)
            e.heard(d["type"], d.get("also"))
            self.items[eid] = e
            out[eid] = e
        return out
