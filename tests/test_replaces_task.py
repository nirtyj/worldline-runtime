"""A request System 1 confidently marks replaces_task cancels the task in hand, like a correction.

    .venv-thor/bin/python -m unittest tests.test_replaces_task
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.harness import Runtime  # noqa: E402
from agent.state import TaskState, TraceLog  # noqa: E402


class FakeHandle:
    def __init__(self) -> None:
        self.created_for, self.done, self.skill = 1, False, "navigate"
        self.resources = {"base"}
        self.canceled = False

    def cancel(self) -> None:
        self.canceled = True


def runtime(busy: bool) -> tuple[Runtime, FakeHandle]:
    """Just the parts of a Runtime that labelling and applying a message touch."""
    rt = object.__new__(Runtime)
    clock = SimpleNamespace(now=lambda: 10.0)
    rt.clock = clock
    rt.task = TaskState(intent_version=1, goal="Bring me a book.")
    rt.tracer = TraceLog(clock)
    rt.state = SimpleNamespace(record=lambda *a, **k: None)
    rt.speech = SimpleNamespace(busy=lambda: False, drop_older_than=lambda v: [])
    rt.belief = SimpleNamespace(mark_hand_unknown=lambda *a: None)
    rt._directives_by_utterance, rt._canceled = {}, []
    rt._start_reconcile = lambda: None
    handle = FakeHandle()
    rt.actions = {1: handle} if busy else {}
    return rt, handle


def utterance(text: str, **directive: object) -> SimpleNamespace:
    d = {"text": text, "kind": "request", **directive} if directive else None
    return SimpleNamespace(id="u2", t_end=10.0, text=text, directive=d)


TEXT = "Never mind the book, get me the alarm clock."


class ReplacesTask(unittest.TestCase):
    def test_directive_keeps_the_flag(self) -> None:
        rt, _ = runtime(busy=True)
        d = rt._directive_for(utterance(TEXT, source="system1", confidence=0.9, replaces_task=True), "request")
        self.assertTrue(d.replaces_task)
        self.assertTrue(d.to_dict()["replaces_task"])

    def test_confident_system1_request_while_busy_cancels_like_a_correction(self) -> None:
        rt, handle = runtime(busy=True)
        rt._apply(utterance(TEXT, source="system1", confidence=0.9, replaces_task=True), "request")
        self.assertTrue(handle.canceled)
        self.assertEqual(rt.task.intent_version, 2)
        self.assertEqual(rt.task.goal, TEXT)
        types = [r["type"] for r in rt.tracer.rows]
        self.assertIn("replaces_task", types)
        self.assertIn("correction", types)

    def test_unsure_label_does_not_cancel(self) -> None:
        rt, handle = runtime(busy=True)
        rt._apply(utterance(TEXT, source="system1", confidence=0.3, replaces_task=True), "request")
        self.assertFalse(handle.canceled)
        self.assertEqual(rt.task.intent_version, 1)

    def test_planner_label_does_not_cancel(self) -> None:
        rt, handle = runtime(busy=True)
        rt._apply(utterance(TEXT), "request")             # no System 1: the planner labelled it
        self.assertFalse(handle.canceled)

    def test_request_that_adds_to_the_task_does_not_cancel(self) -> None:
        rt, handle = runtime(busy=True)
        rt._apply(utterance("And a cup too.", source="system1", confidence=0.9, replaces_task=False), "request")
        self.assertFalse(handle.canceled)
        self.assertEqual(rt.task.intent_version, 1)

    def test_idle_request_is_just_a_request(self) -> None:
        rt, _ = runtime(busy=False)
        rt._apply(utterance(TEXT, source="system1", confidence=0.9, replaces_task=True), "request")
        self.assertEqual(rt.task.intent_version, 2)
        self.assertNotIn("correction", [r["type"] for r in rt.tracer.rows])


if __name__ == "__main__":
    unittest.main()
