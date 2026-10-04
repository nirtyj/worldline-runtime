"""The runtime never reaches past the robot API: no world, no simulator, no perception internals.

agent/, brains/ and baseline/ get what the robot reports (sim/robot.py) and nothing
else. The world knows the truth (sim/world.py), so importing it, or a simulator, from
the runtime would let truth leak into belief.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNTIME = ("agent", "brains", "baseline")
FORBIDDEN = ("thor", "ai2thor", "sim.world", "sim.robot", "sim.layout", "perception", "wlsim")


def imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


class Boundaries(unittest.TestCase):
    def test_runtime_never_imports_a_world(self) -> None:
        bad = []
        for pkg in RUNTIME:
            for path in sorted((ROOT / pkg).rglob("*.py")):
                for name in imports(path):
                    if any(name == f or name.startswith(f + ".") for f in FORBIDDEN):
                        bad.append(f"{path.relative_to(ROOT)}: {name}")
        self.assertEqual(bad, [])

    def test_the_check_sees_a_forbidden_import(self) -> None:
        tmp = ROOT / "runs" / "boundary_probe.py"
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_text("from sim.world import World\nimport thor.world\n")
        try:
            self.assertTrue({"sim.world", "thor.world"} <= imports(tmp))
        finally:
            tmp.unlink()


if __name__ == "__main__":
    unittest.main()
