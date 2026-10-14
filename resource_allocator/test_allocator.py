from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from allocator import (
    AllocationInputError,
    allocate_from_file,
    allocate_resources,
    summarize,
)


@pytest.fixture
def four_zones() -> list[dict]:
    return [
        {"zone_id": "ZONE_B", "priority_score": 84.5, "estimated_travel_minutes": 12.0},
        {"zone_id": "ZONE_C", "priority_score": 71.0, "estimated_travel_minutes": 18.0},
        {"zone_id": "ZONE_D", "priority_score": 58.0, "estimated_travel_minutes": 25.0},
        {"zone_id": "ZONE_A", "priority_score": 14.0, "estimated_travel_minutes": 8.0},
    ]


# Test 1: 3 teams, 4 zones -> 3 ASSIGNED, 1 WAITING
def test_01_three_teams_four_zones(four_zones: list[dict]) -> None:
    res = allocate_resources(four_zones, team_count=3)
    assert len(res) == 4
    assert [r["assignment_status"] for r in res] == [
        "ASSIGNED",
        "ASSIGNED",
        "ASSIGNED",
        "WAITING",
    ]
    assert [r["team_id"] for r in res] == ["TEAM_01", "TEAM_02", "TEAM_03", None]
    assert [r["assigned_zone_id"] for r in res] == ["ZONE_B", "ZONE_C", "ZONE_D", "ZONE_A"]
    assert res[0]["reason"] == "Highest-priority zone (rank 1 of 4, score 84.5)"
    assert res[1]["reason"] == "Next-highest priority zone (rank 2 of 4, score 71.0)"
    assert res[3]["reason"] == "Waiting: all 3 teams are on higher-priority zones (rank 4 of 4)"


# Test 2: 0 teams -> all WAITING
def test_02_zero_teams_all_waiting(four_zones: list[dict]) -> None:
    res = allocate_resources(four_zones, team_count=0)
    assert len(res) == 4
    assert all(r["assignment_status"] == "WAITING" for r in res)
    assert all(r["team_id"] is None for r in res)
    assert all(r["reason"] == "Waiting: no response teams available" for r in res)


# Test 3: More teams than zones -> no duplicate zones, summarize reports idle teams
def test_03_more_teams_than_zones(four_zones: list[dict]) -> None:
    res = allocate_resources(four_zones, team_count=6)
    assert len(res) == 4
    assert all(r["assignment_status"] == "ASSIGNED" for r in res)
    summary = summarize(res)
    assert summary["assigned_count"] == 4
    assert summary["waiting_count"] == 0
    assert summary["idle_team_count"] == 2
    assert summary["available_team_count"] == 6
    assert summary["unmet_zone_ids"] == []


# Test 4: Equal scores -> deterministic tie-break by shorter travel time
def test_04_tie_break_by_travel_time() -> None:
    zones = [
        {"zone_id": "ZONE_C", "priority_score": 60.0, "estimated_travel_minutes": 22.0},
        {"zone_id": "ZONE_B", "priority_score": 60.0, "estimated_travel_minutes": 11.0},
    ]
    res = allocate_resources(zones, team_count=2)
    assert [r["assigned_zone_id"] for r in res] == ["ZONE_B", "ZONE_C"]
    assert "(tied score 60.0; shorter supplied travel time)" in res[0]["reason"]


# Test 5: Equal scores without complete travel time -> tie-break by zone_id alphabetically
def test_05_tie_break_by_zone_id() -> None:
    zones = [
        {"zone_id": "ZONE_C", "priority_score": 60.0},
        {"zone_id": "ZONE_B", "priority_score": 60.0},
    ]
    res = allocate_resources(zones, team_count=1)
    assert [r["assigned_zone_id"] for r in res] == ["ZONE_B", "ZONE_C"]
    assert "(tied score 60.0 with ZONE_C; ZONE_B first by zone_id)" in res[0]["reason"]
    assert "Waiting: team is on higher-priority zones (rank 2 of 2)" in res[1]["reason"]


# Test 6: Input list and dicts are never mutated
def test_06_input_immutability(four_zones: list[dict]) -> None:
    original_copy = copy.deepcopy(four_zones)
    _ = allocate_resources(four_zones, team_count=2)
    assert four_zones == original_copy


# Test 7: Exact unrounded priority_score and required output keys preserved
def test_07_exact_keys_and_unrounded_score() -> None:
    zones = [{"zone_id": "ZONE_X", "priority_score": 84.56789}]
    res = allocate_resources(zones, team_count=1)
    required_keys = {
        "team_id",
        "assigned_zone_id",
        "priority_score",
        "assignment_status",
        "reason",
        "available_team_count",
        "rank",
        "source",
    }
    assert required_keys.issubset(res[0].keys())
    assert res[0]["priority_score"] == 84.56789
    assert res[0]["source"] == "priority_first_baseline"


# Test 8: Input reason note appended properly
def test_08_input_reason_note_appended() -> None:
    zones = [
        {
            "zone_id": "ZONE_B",
            "priority_score": 90.0,
            "reason": "Hospital road submerged",
        }
    ]
    res = allocate_resources(zones, team_count=1)
    assert res[0]["reason"].endswith(" | Zone note: Hospital road submerged")


# Test 9: Empty zone list [] returns [] cleanly
def test_09_empty_zone_list() -> None:
    res = allocate_resources([], team_count=3)
    assert res == []
    summary = summarize(res)
    assert summary["assigned_count"] == 0
    assert summary["idle_team_count"] == 0


# Test 10: Missing, blank, or duplicate zone_id raises AllocationInputError
@pytest.mark.parametrize(
    "bad_zones",
    [
        [{"priority_score": 50.0}],
        [{"zone_id": "   ", "priority_score": 50.0}],
        [
            {"zone_id": "ZONE_A", "priority_score": 50.0},
            {"zone_id": "ZONE_A", "priority_score": 60.0},
        ],
    ],
)
def test_10_invalid_or_duplicate_zone_id(bad_zones: list[dict]) -> None:
    with pytest.raises(AllocationInputError):
        allocate_resources(bad_zones, team_count=1)


# Test 11: Invalid priority_score (bool, text, NaN, Inf, <0, >100) raises AllocationInputError
@pytest.mark.parametrize(
    "bad_score",
    [None, True, False, "80.0", float("nan"), float("inf"), -0.1, 100.1],
)
def test_11_invalid_priority_score(bad_score: object) -> None:
    with pytest.raises(AllocationInputError):
        allocate_resources([{"zone_id": "ZONE_1", "priority_score": bad_score}], team_count=1)


# Test 12: Invalid team_count (negative, float, bool, text) raises AllocationInputError
@pytest.mark.parametrize("bad_teams", [-1, 2.5, True, False, "3", None])
def test_12_invalid_team_count(bad_teams: object, four_zones: list[dict]) -> None:
    with pytest.raises(AllocationInputError):
        allocate_resources(four_zones, team_count=bad_teams)  # type: ignore[arg-type]


# Test 13: 100+ teams uses 3-digit formatting (TEAM_001 ... TEAM_100)
def test_13_hundred_teams_formatting(four_zones: list[dict]) -> None:
    res = allocate_resources(four_zones, team_count=100)
    assert res[0]["team_id"] == "TEAM_001"
    assert res[1]["team_id"] == "TEAM_002"


# Test 14: File I/O supports JSON list, JSON object with "zones", and CSV
def test_14_allocate_from_file_json_and_csv(tmp_path: Path, four_zones: list[dict]) -> None:
    json_obj_path = tmp_path / "zones_wrapped.json"
    json_obj_path.write_text(json.dumps({"zones": four_zones}), encoding="utf-8")

    out_csv_path = tmp_path / "out.csv"
    res = allocate_from_file(str(json_obj_path), team_count=2, output_path=str(out_csv_path))
    assert len(res) == 4
    assert out_csv_path.exists()
    assert "Waiting: both teams are on higher-priority zones (rank 3 of 4)" in res[2]["reason"]


# Test 15: Deterministic output regardless of input ordering
def test_15_deterministic_shuffled_order(four_zones: list[dict]) -> None:
    shuffled = [four_zones[3], four_zones[1], four_zones[0], four_zones[2]]
    res1 = allocate_resources(four_zones, team_count=2)
    res2 = allocate_resources(shuffled, team_count=2)
    assert res1 == res2