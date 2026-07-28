"""Validated, deterministic research slice specifications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_research_slices(path: Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("slices"), list):
        raise ValueError("research slice spec must be schema_version=1 with a slices list")
    result: list[dict[str, object]] = []
    names: set[str] = set()
    ranges: list[tuple[int, int]] = []
    for item in payload["slices"]:
        if not isinstance(item, dict):
            raise ValueError("slice entries must be objects")
        name = item.get("name")
        offset = item.get("offset")
        count = item.get("count")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("slice names must be non-empty and unique")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"invalid offset for slice {name}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"invalid count for slice {name}")
        end = offset + count
        if any(offset < other_end and other_start < end for other_start, other_end in ranges):
            raise ValueError(f"overlapping slice: {name}")
        names.add(name)
        ranges.append((offset, end))
        result.append({"name": name, "offset": offset, "count": count, "purpose": str(item.get("purpose", ""))})
    if not result:
        raise ValueError("research slice spec must contain at least one slice")
    return result


def resolve_research_slice(path: Path, name: str, *, manifest_length: int | None = None) -> dict[str, object]:
    matches = [item for item in load_research_slices(path) if item["name"] == name]
    if len(matches) != 1:
        raise KeyError(f"unknown research slice: {name}")
    item = matches[0]
    if manifest_length is not None and int(item["offset"]) + int(item["count"]) > manifest_length:
        raise ValueError(f"slice exceeds manifest length: {name}")
    return item
