"""Perception: how the robot (sim/robot.py) learns what is in front of it.

source.py is the contract and the stand-in detector (the world's own object list,
limited to what the head camera sees). The server picks one with
``--perception module:factory``, where factory(world) returns the source.
"""

from .source import PerceptionSource, StandIn, create

__all__ = ["PerceptionSource", "StandIn", "create"]
