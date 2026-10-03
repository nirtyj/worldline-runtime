"""Souls: who an agent is, and what it wants to do when nobody is asking.

A soul is one thing, the persona: a markdown file (personas/<name>.md) whose prose makes
the agent who it is, with a short header for what the code needs, and its own LLM, which
reads that prose and decides what the agent wants to do next. The robot's is
personas/robot.md.

Header (between --- lines, one "key: value" per line):
  role: mother
  autonomy: navigate, look, pick, place, say   tools its own goals may use (a list: commas)
  cadence_s: 30                                 how often it thinks about what it wants, at most
  habits: mug=kitchen, book=bedroom             a mapping: object type -> the room it belongs in
  anything else (rooms, steps_aside, honesty ...) is read by the LLM with the prose

SoulPersona is a drop-in for agent/persona.py's Persona: the runtime calls the same
propose / is_done / finish / heard / snapshot. propose() is called every half second and
must not block, so it starts the LLM call in the background and hands the resulting goal
over on a later call. A soul's goal ends when the planner has nothing left to do for it
(done_on_wait), or after goal_timeout_s.
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .persona import IDLE_S, OwnGoal

SOULS = Path(__file__).resolve().parents[1] / "personas"
DEFAULT_TOOLS = ("navigate", "look", "say")
SOUL_TIMEOUT_S = 20.0

SOUL_SYSTEM = """You are the inner voice of {name}. You decide what {name} wants to do next, when \
nobody is asking {them} for anything. Read WHO YOU ARE and stay in character: a tidy person tidies, \
a nine-year-old plays, a robot keeps the home known. A separate planner carries your intention out \
step by step with the body, which can: {abilities}.

Call exactly one tool:
- intend: what you want to do now, as one concrete sentence you'd say to yourself, naming the objects, \
people and places from BELIEF and PEOPLE ("put mug_1 back in the kitchen", "ask the robot where \
book_2 is", "look at the bedroom dresser again"). Only use ids you believe exist. Optionally a line \
to say out loud first, and to whom.
- rest: nothing you want to do right now, and for how long.

Don't repeat what you just did (WHAT YOU DID) unless it failed and trying again makes sense."""

INTEND_TOOL = {
    "name": "intend",
    "description": "What you want to do now.",
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string", "description": "The intention, one concrete sentence."},
        "say": {"type": "string", "description": "A line to say out loud first (optional)."},
        "to": {"type": "string", "description": "Who that line is for, by name (optional)."},
    }, "required": ["text"]},
}
REST_TOOL = {
    "name": "rest",
    "description": "Nothing to do right now.",
    "parameters": {"type": "object", "properties": {
        "seconds": {"type": "number", "description": "How long before thinking about it again (10-300)."},
        "why": {"type": "string", "description": "Why, in a few words."},
    }, "required": ["seconds"]},
}


@dataclass
class Soul:
    name: str
    meta: dict[str, Any]
    prose: str

    @property
    def role(self) -> str:
        return str(self.meta.get("role") or self.name)

    @property
    def autonomy(self) -> tuple[str, ...]:
        """The tools its own goals may use. Picking brings what the harness requires before a
        pick: a look and a reachability check."""
        a = self.meta.get("autonomy")
        items = [str(t) for t in (a if isinstance(a, list) else [a] if a else list(DEFAULT_TOOLS))]
        if "pick" in items or "place" in items:
            items += ["look", "reachability"]
        return tuple(dict.fromkeys(items))

    @property
    def cadence_s(self) -> float:
        try:
            return float(self.meta.get("cadence_s", 30))
        except (TypeError, ValueError):
            return 30.0

    @property
    def habits(self) -> dict[str, str]:
        h = self.meta.get("habits") or {}
        return h if isinstance(h, dict) else {}


def parse_soul(name: str, text: str) -> Soul:
    meta: dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        head, _, body = text[3:].partition("\n---")
        for line in head.strip().splitlines():
            key, sep, value = line.partition(":")
            if sep:
                meta[key.strip()] = _value(value.strip())
    return Soul(name, meta, body.strip())


def _value(v: str) -> Any:
    items = [x.strip() for x in v.split(",") if x.strip()]
    if items and all("=" in x for x in items):
        return {k.strip(): val.strip() for k, _, val in (x.partition("=") for x in items)}
    if len(items) > 1:
        return items
    return v


def load_soul(name: str) -> Soul:
    return parse_soul(name, (SOULS / f"{name}.md").read_text())


# A custom soul, written by the LLM from a sentence or two on the page ("a gardening robot that
# tells you when a plant looks dry"), in the same format as personas/*.md.
ROBOT_TOOLS = ("navigate", "look", "pick", "place", "say")
SOUL_WRITER_SYSTEM = """You write the soul of a home robot in a simulated house, from a short \
description. The robot has a mobile base, two arms and a speaker: it can navigate between spots, \
look, pick objects up, place them, and say things. Its soul decides what it does on its own when \
nobody is asking; a planner and a runtime with fixed safety rules carry that out, so the soul can't \
change those rules. Write the robot the description asks for, in character, but only what this robot \
can really do in a home. Whoever is asking always comes first, and it never takes something out of \
someone's hands. Call write_soul once."""
WRITE_SOUL_TOOL = {
    "name": "write_soul",
    "description": "The robot's soul, written from the description.",
    "parameters": {"type": "object", "properties": {
        "role": {"type": "string", "description": "A few words: what kind of robot this is, e.g. 'gardening robot'."},
        "autonomy": {"type": "array", "items": {"type": "string", "enum": list(ROBOT_TOOLS)},
                     "description": "The tools its own goals may use. Leave out pick and place if it should never "
                                    "move things, and say if it should stay silent."},
        "cadence_s": {"type": "number", "description": "How often it thinks about what to do when idle, in "
                                                       "seconds (15-300); lower is busier."},
        "habits": {"type": "string", "description": "Where things belong, if it tidies: 'mug=kitchen, "
                                                    "book=bedroom' (object type=room). Empty if it doesn't move things."},
        "prose": {"type": "string", "description": "Two or three short paragraphs in the first person: who it is, "
                                                   "what it does when nobody needs it, and how it talks."},
    }, "required": ["role", "autonomy", "cadence_s", "prose"]},
}


async def write_soul(client: Any, description: str) -> str:
    """A soul file (header and prose) written from a short description by the soul's LLM."""
    description = description.strip()
    if not description:
        raise ValueError("describe the robot first")
    use = await asyncio.wait_for(client.tool_call(SOUL_WRITER_SYSTEM, f"DESCRIPTION\n{description[:2000]}",
                                                  [WRITE_SOUL_TOOL]), SOUL_TIMEOUT_S)
    if use.name != "write_soul":
        raise ValueError("the model didn't write a soul; try a different description")
    a = dict(use.args)
    role = " ".join(str(a.get("role") or "robot").replace(",", " ").split())[:60]   # a comma would make it a list
    tools = [t for t in ROBOT_TOOLS if t in (a.get("autonomy") or [])] or list(DEFAULT_TOOLS)
    try:
        cadence = max(15.0, min(300.0, float(a.get("cadence_s") or 30)))
    except (TypeError, ValueError):
        cadence = 30.0
    habits = ", ".join(x.strip() for x in str(a.get("habits") or "").replace("\n", ",").split(",")
                       if x.count("=") == 1 and all(p.strip() for p in x.split("=")))
    prose = "\n".join(line for line in str(a.get("prose") or "").strip().splitlines() if line.strip() != "---")
    head = ["---", f"role: {role}", f"autonomy: {', '.join(tools)}", f"cadence_s: {cadence:g}"]
    head += [f"habits: {habits}"] if habits else []
    head += ["steps_aside: yes", "honesty: truthful", "---"]
    return "\n".join(head + [f"# The {role}", "", prose, ""])


class SoulPersona:
    done_on_wait = True           # the goal is done when the planner has nothing left to do for it
    goal_timeout_s = 240.0

    def __init__(self, soul: Soul, client: Any, name: str | None = None, level: str = "on",
                 log_path: Path | None = None) -> None:
        self.soul, self.client = soul, client
        self.log_path = log_path                 # every call, appended as text (for reading a run afterwards)
        self.name = name or soul.name
        self.level = level
        self.goal: OwnGoal | None = None
        self.permission = "granted"
        self.last: list[tuple[float, str, str]] = []
        self.history: collections.deque[dict[str, Any]] = collections.deque(maxlen=12)
        self.calls: collections.deque[dict[str, Any]] = collections.deque(maxlen=20)   # for the page
        self.finished_t = -1e9
        self._ids = itertools.count(1)
        self._task: asyncio.Task | None = None
        self._ready: dict[str, Any] | None = None
        self._next_t = 0.0
        self._injected: list[str] = []
        self.runtime: Any = None
        self.tokens_in = self.tokens_out = 0

    def bind(self, runtime: Any) -> None:
        """The runtime this soul belongs to: what it has heard and done, for the prompt."""
        self.runtime = runtime

    @property
    def acts(self) -> bool:
        return self.level not in ("off", "quiet")

    def heard(self, text: str, now: float, reply: str | None = None) -> bool:
        return False

    def tick(self, now: float) -> None:
        pass

    def inject(self, text: str) -> None:
        """Put an intention in this agent's head, as if it had thought of it."""
        self._injected.append(text)

    # ------------------------------------------------------------------
    def propose(self, belief: Any, map_: dict[str, Any], now: float, quiet_s: float) -> OwnGoal | None:
        if self.goal is not None:
            return None
        if self._injected:              # a scenario's intention runs even when quiet: it isn't the soul's own
            return self._start({"text": self._injected.pop(0)}, now, drive="scenario")
        if not self.acts:
            return None
        if self._ready is not None:
            ready, self._ready = self._ready, None
            return self._start(ready, now, drive="soul")
        if quiet_s < IDLE_S or now < self._next_t or (self._task is not None and not self._task.done()):
            return None
        self._next_t = now + self.soul.cadence_s
        self._task = asyncio.create_task(self._think(self.render(belief, map_, now), now))
        return None

    def _start(self, intent: dict[str, Any], now: float, drive: str) -> OwnGoal:
        text = str(intent["text"]).strip()
        tools = self.soul.autonomy
        if intent.get("say"):
            to = f" to {intent['to']}" if intent.get("to") else ""
            text = f"First say{to}: \"{intent['say']}\". Then: {text}"
            tools = tuple(dict.fromkeys((*tools, "say")))
        self.goal = OwnGoal(f"g{next(self._ids)}", drive, str(intent.get("to") or ""), text, tools, round(now, 2))
        return self.goal

    async def _think(self, prompt: str, now: float) -> None:
        system = SOUL_SYSTEM.format(name=self.name, them="it" if self.name == "robot" else "them",
                                    abilities=", ".join(self.soul.autonomy))
        t0 = time.monotonic()
        try:
            use = await asyncio.wait_for(self.client.tool_call(system, prompt, [INTEND_TOOL, REST_TOOL]), SOUL_TIMEOUT_S)
        except Exception as e:                         # a soul that can't think just rests a while
            self.calls.append({"t": round(now, 1), "error": repr(e)[:200]})
            return
        self.tokens_in += getattr(use, "input_tokens", 0) or 0
        self.tokens_out += getattr(use, "output_tokens", 0) or 0
        call = {"t": round(now, 1), "tool": use.name, "args": dict(use.args), "latency_s": round(time.monotonic() - t0, 2),
                "prompt": prompt}
        self.calls.append(call)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(f"==== t={now:.1f} {use.name} {use.args} ({call['latency_s']} s)\n{prompt}\n\n")
        if use.name == "intend" and str(use.args.get("text", "")).strip():
            self._ready = dict(use.args)
        elif use.name == "rest":
            try:
                seconds = max(10.0, min(300.0, float(use.args.get("seconds", 60))))
            except (TypeError, ValueError):
                seconds = 60.0
            self._next_t = now + seconds

    def render(self, belief: Any, map_: dict[str, Any], now: float) -> str:
        """What the soul's LLM reads: who it is, where it is, what it believes, what it heard and did."""
        kps = map_.get("keypoints") or {}
        at = belief.robot_at.value
        room = (kps.get(at) or {}).get("room") if at else None
        held = [f"{b.value} ({arm} hand)" for arm, b in belief.holding.items() if b.value]
        lines = [f"TIME {now:.0f} s", f"YOU ARE {self.name} ({self.soul.role}), at {at or 'somewhere'}"
                 + (f" in the {room}" if room else "") + (f", holding {', '.join(held)}" if held else "")]
        objs = []
        for oid, ob in sorted(belief.objects.items()):
            where = ob.where.value
            if where in (None, "UNKNOWN"):
                objs.append(f"  {oid}: don't know where")
                continue
            r = (kps.get(str(where)) or {}).get("room")
            src = "saw it" if ob.where.verified else {"memory": "remember", "prior": "knew",
                                                      "skill": "put it there"}.get(ob.where.source, ob.where.source)
            objs.append(f"  {oid}: {where}" + (f" ({r})" if r and r not in str(where) else "")
                        + f" [{src}, {max(0, now - ob.where.t):.0f} s ago]")
        lines.append("BELIEF (where you think things are)\n" + ("\n".join(objs[:60]) or "  nothing yet"))
        people = getattr(belief, "people", {}) or {}
        seen = [f"  {p}: near {info.get('keypoint')}" + (f", {now - info['t']:.0f} s ago" if info.get("t") is not None else "")
                for p, info in sorted(people.items()) if p != self.name]
        lines.append("PEOPLE (where you last saw them)\n" + ("\n".join(seen) or "  nobody seen yet"))
        rt = self.runtime
        if rt is not None:
            heard = [f"  {getattr(u, 'speaker', 'user')}"
                     + (f" to {u.to}" if getattr(u, "to", None) else "") + f": {u.text}"
                     for u in rt.task.utterances[-8:]]
            said = [f"  you said: {e.args.get('text')}" for e in rt.history[-20:] if e.tool == "say"
                    and e.status not in ("DROPPED", "REJECTED")][-4:]
            lines.append("RECENT CONVERSATION\n" + ("\n".join(heard + said) or "  nothing"))
        done = [f"  {h['text'][:90]} -> {h['outcome']}" + (f" ({h['detail']})" if h.get("detail") else "")
                for h in list(self.history)[-6:]]
        lines.append("WHAT YOU DID\n" + ("\n".join(done) or "  nothing yet"))
        habits = self.soul.habits
        if habits:
            lines.append("WHERE THINGS BELONG (your habits)\n  " + ", ".join(f"{k}: {v}" for k, v in habits.items()))
        lines.append("WHO YOU ARE\n" + self.soul.prose)
        lines.append("What do you want to do now? Call intend or rest.")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    def is_done(self, belief: Any) -> bool:
        return False                                   # the planner says when (done_on_wait)

    def finish(self, outcome: str, now: float, detail: str | None = None) -> OwnGoal | None:
        g, self.goal = self.goal, None
        if g is None:
            return None
        self.finished_t = now
        self._next_t = max(self._next_t, now + min(10.0, self.soul.cadence_s))
        self.history.append({"id": g.id, "drive": g.drive, "target": g.target, "text": g.text, "outcome": outcome,
                             "detail": detail, "t": round(now, 1), "took_s": round(now - g.t_start, 1)})
        return g

    def snapshot(self, now: float) -> dict[str, Any]:
        g = self.goal
        return {
            "name": self.name, "level": self.level, "permission": self.permission, "enabled": self.acts,
            "soul": self.soul.role,
            "goal": None if g is None else {**g.to_dict(), "age_s": round(now - g.t_start, 1)},
            "drives": [], "history": list(self.history)[::-1],
            "thinking": self._task is not None and not self._task.done(),
            "last_call": (lambda c: {k: v for k, v in c.items() if k != "prompt"} if c else None)(self.calls[-1] if self.calls else None),
        }
