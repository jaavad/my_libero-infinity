"""Utilities for auditing generated perturbations and displacement metrics."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any

from libero_infinity.task_config import (
    _IMMOBILE_WORKSPACE_FIXTURES,
    FixtureInfo,
    ObjectInfo,
    TaskConfig,
    _support_parent_names,
)

_HARD_REQUIRE_RE = re.compile(r"^\s*require(?!\[)\s+(?P<body>.+)$")
_SOFT_REQUIRE_RE = re.compile(r"^\s*require\[(?P<weight>[^\]]+)\]\s+(?P<body>.+)$")
_TEMPORAL_OPERATOR_RE = re.compile(r"\b(always|eventually|until|next|monitor|implies)\b")


@dataclass(frozen=True)
class ConstraintAudit:
    hard_require_total: int = 0
    soft_require_total: int = 0
    hard_axis_clearance: int = 0
    hard_distance_clearance: int = 0
    soft_ood_bias: int = 0
    temporal_require_total: int = 0
    temporal_operators: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NumericSummary:
    count: int
    mean: float | None
    median: float | None
    p10: float | None
    p90: float | None
    minimum: float | None
    maximum: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_generated_constraints(scenic_code: str) -> ConstraintAudit:
    """Classify hard/soft constraints in a generated Scenic program."""
    hard_total = 0
    soft_total = 0
    hard_axis = 0
    hard_distance = 0
    soft_ood = 0
    temporal_total = 0
    temporal_ops: set[str] = set()

    for raw_line in scenic_code.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        soft_match = _SOFT_REQUIRE_RE.match(line)
        if soft_match:
            soft_total += 1
            body = soft_match.group("body")
            if "distance from" in body and "> _ood_margin" in body:
                soft_ood += 1
            matches = set(_TEMPORAL_OPERATOR_RE.findall(body))
            if matches:
                temporal_total += 1
                temporal_ops.update(matches)
            continue

        hard_match = _HARD_REQUIRE_RE.match(line)
        if not hard_match:
            continue

        hard_total += 1
        body = hard_match.group("body")
        if "abs(" in body and (".position.x" in body or ".position.y" in body):
            hard_axis += 1
        elif "distance from" in body:
            hard_distance += 1

        matches = set(_TEMPORAL_OPERATOR_RE.findall(body))
        if matches:
            temporal_total += 1
            temporal_ops.update(matches)

    return ConstraintAudit(
        hard_require_total=hard_total,
        soft_require_total=soft_total,
        hard_axis_clearance=hard_axis,
        hard_distance_clearance=hard_distance,
        soft_ood_bias=soft_ood,
        temporal_require_total=temporal_total,
        temporal_operators=tuple(sorted(temporal_ops)),
    )


def canonical_xy_for_object(cfg: TaskConfig, obj: ObjectInfo) -> tuple[float, float] | None:
    """Return the canonical xy anchor for an object, including contained cases."""
    if obj.init_x is not None and obj.init_y is not None:
        return (float(obj.init_x), float(obj.init_y))

    if not obj.region_name:
        return _fallback_support_anchor_xy(cfg, obj)

    region = cfg.regions.get(obj.region_name)
    if region is None or not region.has_bounds:
        return _fallback_support_anchor_xy(cfg, obj)
    if obj.placement_target and region.target != obj.placement_target:
        return _fallback_support_anchor_xy(cfg, obj)
    centre = region.centre
    if centre is None:
        return _fallback_support_anchor_xy(cfg, obj)
    return (float(centre[0]), float(centre[1]))


def moving_support_names(
    cfg: TaskConfig,
) -> tuple[set[str], set[str], dict[str, str | None]]:
    """Return movable support fixture names, movable support object names, parent map."""
    fixture_by_name = {fixture.instance_name: fixture for fixture in cfg.fixtures}
    support_fixture_names = {
        obj.placement_target
        for obj in cfg.movable_objects
        if obj.placement_target in fixture_by_name
        if fixture_by_name[obj.placement_target].fixture_class not in _IMMOBILE_WORKSPACE_FIXTURES
    }
    moving_fixture_names = cfg.goal_fixture_names | support_fixture_names
    support_parent_map = _support_parent_names(
        cfg.movable_objects,
        moving_fixture_names=moving_fixture_names,
    )
    movable_names = {obj.instance_name for obj in cfg.movable_objects}
    movable_support_names = {
        parent_name for parent_name in support_parent_map.values() if parent_name in movable_names
    }
    return moving_fixture_names, movable_support_names, support_parent_map


def object_displacements(
    cfg: TaskConfig,
    scene_objects: list[Any],
) -> dict[str, float]:
    """Compute xy displacement of each movable object from its canonical pose."""
    positions = _scene_xy_positions(scene_objects)
    displacements: dict[str, float] = {}
    for obj in cfg.movable_objects:
        if obj.instance_name not in positions:
            continue
        canonical_xy = canonical_xy_for_object(cfg, obj)
        if canonical_xy is None:
            continue
        displacements[obj.instance_name] = _xy_distance(positions[obj.instance_name], canonical_xy)
    return displacements


def support_displacements(
    cfg: TaskConfig,
    scene_objects: list[Any],
) -> dict[str, float]:
    """Compute xy displacement of movable support anchors from canonical pose."""
    positions = _scene_xy_positions(scene_objects)
    moving_fixture_names, movable_support_names, _support_parent_map = moving_support_names(cfg)

    displacements: dict[str, float] = {}
    fixture_by_name = {fixture.instance_name: fixture for fixture in cfg.fixtures}
    for fixture_name in moving_fixture_names:
        fixture = fixture_by_name.get(fixture_name)
        if (
            fixture is None
            or fixture.init_x is None
            or fixture.init_y is None
            or fixture_name not in positions
        ):
            continue
        displacements[fixture_name] = _xy_distance(
            positions[fixture_name],
            (float(fixture.init_x), float(fixture.init_y)),
        )

    object_by_name = {obj.instance_name: obj for obj in cfg.movable_objects}
    for object_name in movable_support_names:
        obj = object_by_name.get(object_name)
        if obj is None or object_name not in positions:
            continue
        canonical_xy = canonical_xy_for_object(cfg, obj)
        if canonical_xy is None:
            continue
        displacements[object_name] = _xy_distance(positions[object_name], canonical_xy)

    return displacements


def summarize_numeric(values: list[float]) -> NumericSummary:
    """Return basic summary statistics for a list of floats."""
    if not values:
        return NumericSummary(
            count=0,
            mean=None,
            median=None,
            p10=None,
            p90=None,
            minimum=None,
            maximum=None,
        )

    ordered = sorted(float(value) for value in values)
    return NumericSummary(
        count=len(ordered),
        mean=float(mean(ordered)),
        median=float(median(ordered)),
        p10=_percentile(ordered, 0.10),
        p90=_percentile(ordered, 0.90),
        minimum=float(ordered[0]),
        maximum=float(ordered[-1]),
    )


def fixture_canonical_xy(fixture: FixtureInfo) -> tuple[float, float] | None:
    if fixture.init_x is None or fixture.init_y is None:
        return None
    return (float(fixture.init_x), float(fixture.init_y))


def _scene_xy_positions(scene_objects: list[Any]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for obj in scene_objects:
        libero_name = getattr(obj, "libero_name", "")
        if not libero_name:
            continue
        positions[libero_name] = (float(obj.position.x), float(obj.position.y))
    return positions


def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _fallback_support_anchor_xy(
    cfg: TaskConfig,
    obj: ObjectInfo,
) -> tuple[float, float] | None:
    if obj.placement_target:
        for fixture in cfg.fixtures:
            if fixture.instance_name == obj.placement_target:
                return fixture_canonical_xy(fixture)
        for candidate in cfg.movable_objects:
            if candidate.instance_name == obj.placement_target:
                return canonical_xy_for_object(cfg, candidate)
    if obj.stacked_on:
        for candidate in cfg.movable_objects:
            if candidate.instance_name == obj.stacked_on:
                return canonical_xy_for_object(cfg, candidate)
    return None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty list")
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * q
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return float(values[lo])
    frac = index - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)
