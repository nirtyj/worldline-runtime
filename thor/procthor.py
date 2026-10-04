"""ProcTHOR houses: multi-room homes from the ProcTHOR-10K dataset.

A house is named "procthor-<split>-<index>", e.g. "procthor-train-7". The first
load fetches the dataset with the `prior` package (about 12 s, needs the
network once) and keeps that house as JSON under runs/procthor/, so later loads
are local and instant.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sim.layout import room_at  # noqa: F401  (rooms are named here, found there)

CACHE = Path(__file__).resolve().parents[1] / "runs" / "procthor"
PATTERN = re.compile(r"^procthor-(train|val|test)-(\d+)$")

# Houses offered in the page's room picker: a kitchen plus two or three other rooms,
# small enough to explore in a few minutes.
HOUSES = [       # picked by screening the first houses for reachable spots and usable kitchens
    ("procthor-train-40", "House 40 · kitchen, living room, bedroom (compact)"),
    ("procthor-train-15", "House 15 · kitchen, living room, bedroom, bathroom"),
    ("procthor-train-38", "House 38 · kitchen, living room, bedroom"),
    ("procthor-train-59", "House 59 · kitchen, living room, bedroom, bathroom"),
]

_dataset: Any = None


def is_house(scene: str) -> bool:
    return bool(PATTERN.match(scene))


def load_house(scene: str) -> dict[str, Any]:
    m = PATTERN.match(scene)
    if not m:
        raise ValueError(f"not a ProcTHOR house name: {scene!r} (expected procthor-train-7 and so on)")
    split, index = m.group(1), int(m.group(2))
    path = CACHE / f"{split}-{index}.json"
    if path.exists():
        return json.loads(path.read_text())
    global _dataset
    if _dataset is None:
        import prior                                   # only needed the first time
        _dataset = prior.load_dataset("procthor-10k")
    house = dict(_dataset[split][index])
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(house))
    return house


def rooms(house: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Room name ("kitchen", "bedroom_2") -> its type, floor polygon and centre."""
    counts: dict[str, int] = {}
    for r in house.get("rooms", []):
        counts[r["roomType"]] = counts.get(r["roomType"], 0) + 1
    seen: dict[str, int] = {}
    out = {}
    for r in house.get("rooms", []):
        base = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", r["roomType"]).lower()
        seen[r["roomType"]] = seen.get(r["roomType"], 0) + 1
        name = base if counts[r["roomType"]] == 1 else f"{base}_{seen[r['roomType']]}"
        poly = [(p["x"], p["z"]) for p in r["floorPolygon"]]
        cx = sum(p[0] for p in poly) / len(poly)
        cz = sum(p[1] for p in poly) / len(poly)
        out[name] = {"type": r["roomType"], "label": name.replace("_", " "), "polygon": poly, "center": (cx, cz)}
    return out
