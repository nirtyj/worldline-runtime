"""Baseline harness (v0).

A first version: wait for the user, then ask the brain for one step at a
time and run each step to completion. It works on simple requests.
"""

from .harness import BaselineRuntime


def create_runtime(robot, user, brain, clock) -> BaselineRuntime:
    return BaselineRuntime(robot, user, brain, clock)
