"""The robot's world, backed by AI2-THOR: world.py loads a room and renders it,
robot.py is the robot API the runtime drives."""

from .robot import ThorRobot
from .world import SCENES, ThorWorld

__all__ = ["SCENES", "ThorRobot", "ThorWorld"]
