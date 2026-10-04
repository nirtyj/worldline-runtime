"""The robot's world, backed by AI2-THOR: world.py loads a room and renders it; the robot
API the runtime drives is sim/robot.py (ThorRobot is its old name)."""

from .robot import ThorRobot
from .world import SCENES, ThorWorld


def create_world() -> ThorWorld:
    """The server's default world (``--world thor:create_world``). Simulators left by a
    server that died without stopping them are stopped first."""
    killed = ThorWorld.reap_orphans()
    if killed:
        print(f"[thor] stopped {len(killed)} orphaned simulator(s): {killed}", flush=True)
    return ThorWorld()


__all__ = ["SCENES", "ThorRobot", "ThorWorld", "create_world"]
