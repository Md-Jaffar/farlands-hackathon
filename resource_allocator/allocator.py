from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import math
from typing import Any


class AllocationInputError(ValueError):
    """Raised when zone inputs, fields, or team_count are invalid."""
    pass


def _validate_inputs(zones: Any, team_count: Any) -> None:
    """Validate team_count and zones list strictly without mutating inputs."""
    # 1. Validate team_count (check bool first because bool is a subclass of int in Python!)
    if (
        isinstance(team_count, bool)
        or not isinstance(team_count, int)
        or team_count < 0
    ):
        raise AllocationInputError(
            f"Invalid team_count '{team_count}': must be a non-negative integer."
        )

    # 2. Validate zones container
    if not isinstance(zones, list):
        raise AllocationInputError("Invalid zones input: expected a list of zone dictionaries.")

    seen_zone_ids: set[str] = set()

    for idx, zone in enumerate(zones):
        if not isinstance(zone, dict):
            raise AllocationInputError(f"Invalid zone at index {idx}: expected a dictionary.")

        # 3. Validate zone_id
        if "zone_id" not in zone or zone["zone_id"] is None:
            raise AllocationInputError(f"Missing required field 'zone_id' at index {idx}.")
        
        raw_id = zone["zone_id"]
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise AllocationInputError(
                f"Invalid 'zone_id' at index {idx}: must be a non-empty string."
            )
        
        clean_id = raw_id.strip()
        if clean_id in seen_zone_ids:
            raise AllocationInputError(
                f"Duplicate zone_id '{clean_id}' found: each zone_id must be unique."
            )
        seen_zone_ids.add(clean_id)

        # 4. Validate priority_score
        if "priority_score" not in zone or zone["priority_score"] is None:
            raise AllocationInputError(
                f"Missing required field 'priority_score' for zone '{clean_id}'."
            )
        
        score = zone["priority_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise AllocationInputError(
                f"Invalid 'priority_score' for zone '{clean_id}': must be a numeric value."
            )
        
        if math.isnan(score) or math.isinf(score):
            raise AllocationInputError(
                f"Invalid 'priority_score' for zone '{clean_id}': cannot be NaN or Inf."
            )
        
        if score < 0.0 or score > 100.0:
            raise AllocationInputError(
                f"Invalid 'priority_score' ({score}) for zone '{clean_id}': must be between 0 and 100."
            )

def _has_valid_travel_time(zone: dict[str, Any]) -> bool:
    """Check if a zone has a valid non-negative numeric estimated_travel_minutes."""
    val = zone.get("estimated_travel_minutes")
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    if math.isnan(val) or math.isinf(val) or val < 0:
        return False
    return True


def _format_score(score: int | float) -> str:
    """Format score with at least one decimal place (e.g., 84.5 or 60.0) to match spec templates."""
    if isinstance(score, int) or (isinstance(score, float) and score.is_integer()):
        return f"{float(score):.1f}"
    return str(score)


def _format_team_id(team_num: int, team_count: int) -> str:
    """Format team_id as TEAM_01 (2 digits) or TEAM_100 (3+ digits if team_count >= 100)."""
    width = max(2, len(str(team_count))) if team_count >= 100 else 2
    return f"TEAM_{team_num:0{width}d}"


def allocate_resources(zones: list[dict[str, Any]], team_count: int) -> list[dict[str, Any]]:
    """
    Deterministically assign response teams to highest-priority zones without mutating inputs.
    """
    _validate_inputs(zones, team_count)

    if not zones:
        return []

    # Work on shallow copies so input list/dicts are never mutated
    working_zones = [dict(z) for z in zones]

    # Group zones by exact priority_score
    score_groups: dict[float, list[dict[str, Any]]] = {}
    for z in working_zones:
        score_key = float(z["priority_score"])
        score_groups.setdefault(score_key, []).append(z)

    # Sort groups by priority_score descending and resolve ties per group
    ordered_entries: list[tuple[dict[str, Any], str]] = []
    for score_key in sorted(score_groups.keys(), reverse=True):
        group = score_groups[score_key]
        if len(group) == 1:
            ordered_entries.append((group[0], ""))
        else:
            # Check if EVERY zone in this tied group has a valid non-negative travel time
            all_have_travel = all(_has_valid_travel_time(z) for z in group)
            score_str = _format_score(group[0]["priority_score"])

            if all_have_travel:
                group_sorted = sorted(
                    group,
                    key=lambda z: (
                        float(z["estimated_travel_minutes"]),
                        z["zone_id"].strip(),
                    ),
                )
                first_id = group_sorted[0]["zone_id"].strip()
                # Check if travel times are distinct or also tied
                travel_times = [float(z["estimated_travel_minutes"]) for z in group_sorted]
                if len(set(travel_times)) == len(travel_times):
                    tie_suffix = f" (tied score {score_str}; shorter supplied travel time)"
                else:
                    other_ids = [
                        z["zone_id"].strip()
                        for z in group_sorted
                        if z["zone_id"].strip() != first_id
                    ]
                    others_str = ", ".join(other_ids)
                    tie_suffix = (
                        f" (tied score {score_str} with {others_str}; "
                        f"{first_id} first by zone_id)"
                    )
                for z in group_sorted:
                    ordered_entries.append((z, tie_suffix))
            else:
                group_sorted = sorted(group, key=lambda z: z["zone_id"].strip())
                first_id = group_sorted[0]["zone_id"].strip()
                other_ids = [
                    z["zone_id"].strip()
                    for z in group_sorted
                    if z["zone_id"].strip() != first_id
                ]
                others_str = ", ".join(other_ids)
                for z in group_sorted:
                    tie_suffix = (
                        f" (tied score {score_str} with {others_str}; "
                        f"{first_id} first by zone_id)"
                    )
                    ordered_entries.append((z, tie_suffix))

    total_zones = len(ordered_entries)
    results: list[dict[str, Any]] = []

    for idx, (zone, tie_suffix) in enumerate(ordered_entries):
        rank = idx + 1
        zone_id_clean = zone["zone_id"].strip()
        raw_score = zone["priority_score"]  # Exact unrounded copy
        score_str = _format_score(raw_score)

        if idx < team_count:
            team_id: str | None = _format_team_id(idx + 1, team_count)
            status = "ASSIGNED"
            if rank == 1:
                base_reason = f"Highest-priority zone (rank 1 of {total_zones}, score {score_str})"
            else:
                base_reason = (
                    f"Next-highest priority zone (rank {rank} of {total_zones}, score {score_str})"
                )
            reason_str = base_reason + tie_suffix
        else:
            team_id = None
            status = "WAITING"
            if team_count == 0:
                reason_str = "Waiting: no response teams available" + tie_suffix
            elif team_count == 1:
                reason_str = (
                    f"Waiting: team is on higher-priority zones (rank {rank} of {total_zones})"
                    + tie_suffix
                )
            elif team_count == 2:
                reason_str = (
                    f"Waiting: both teams are on higher-priority zones (rank {rank} of {total_zones})"
                    + tie_suffix
                )
            else:
                reason_str = (
                    f"Waiting: all {team_count} teams are on higher-priority zones "
                    f"(rank {rank} of {total_zones})" + tie_suffix
                )

        # Append input zone note if supplied and non-empty
        input_note = zone.get("reason")
        if isinstance(input_note, str) and input_note.strip():
            reason_str += f" | Zone note: {input_note.strip()}"

        row: dict[str, Any] = {
            "team_id": team_id,
            "assigned_zone_id": zone_id_clean,
            "priority_score": raw_score,
            "assignment_status": status,
            "reason": reason_str,
            "available_team_count": team_count,
            "rank": rank,
            "source": "priority_first_baseline",
        }
        results.append(row)

    return results


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute summary statistics from the output rows of allocate_resources.
    """
    if not isinstance(rows, list):
        raise AllocationInputError("summarize expects a list of allocation output row dictionaries.")

    if not rows:
        return {
            "assigned_count": 0,
            "waiting_count": 0,
            "unassigned_count": 0,
            "idle_team_count": 0,
            "available_team_count": 0,
            "unmet_zone_ids": [],
        }

    assigned_count = sum(1 for r in rows if r.get("assignment_status") == "ASSIGNED")
    waiting_count = sum(1 for r in rows if r.get("assignment_status") == "WAITING")
    unassigned_count = sum(
        1 for r in rows if r.get("assignment_status") in ("WAITING", "UNASSIGNED")
    )
    available_team_count = int(rows[0].get("available_team_count", 0))
    idle_team_count = max(0, available_team_count - assigned_count)
    unmet_zone_ids = [
        r["assigned_zone_id"]
        for r in rows
        if r.get("assignment_status") in ("WAITING", "UNASSIGNED")
    ]

    return {
        "assigned_count": assigned_count,
        "waiting_count": waiting_count,
        "unassigned_count": unassigned_count,
        "idle_team_count": idle_team_count,
        "available_team_count": available_team_count,
        "unmet_zone_ids": unmet_zone_ids,
    }

def _resolve_path(path_str: str) -> Path:
    """Resolve path directly or relative to this script's directory (Rule 3)."""
    p = Path(path_str)
    if p.exists() or p.is_absolute():
        return p
    alt = Path(__file__).resolve().parent / p
    return alt if alt.exists() else p


def _parse_csv_number(val: str | None) -> Any:
    """Convert CSV string values for numeric fields into float/int where possible."""
    if val is None:
        return None
    s = val.strip()
    if s == "":
        return None
    try:
        f = float(s)
        return f
    except ValueError:
        return val


def allocate_from_file(
    zones_path: str, team_count: int, output_path: str | None = None
) -> list[dict[str, Any]]:
    """
    Read zones from a JSON (list or {'zones': [...]}) or CSV file, allocate teams,
    and optionally write the output to JSON or CSV by file extension.
    """
    if not isinstance(zones_path, str) or not zones_path.strip():
        raise AllocationInputError("Invalid zones_path: must be a non-empty file path string.")

    resolved_in = _resolve_path(zones_path)
    if not resolved_in.exists() or not resolved_in.is_file():
        raise AllocationInputError(f"Input file not found: '{zones_path}'.")

    suffix = resolved_in.suffix.lower()
    zones_list: list[dict[str, Any]] = []

    try:
        if suffix == ".json":
            with resolved_in.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                zones_list = data
            elif isinstance(data, dict) and isinstance(data.get("zones"), list):
                zones_list = data["zones"]
            else:
                raise AllocationInputError(
                    f"JSON file '{zones_path}' must contain a list of zones or an object with a 'zones' key."
                )
        elif suffix == ".csv":
            with resolved_in.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise AllocationInputError(f"CSV file '{zones_path}' has no header columns.")
                for row in reader:
                    parsed_row: dict[str, Any] = dict(row)
                    if "priority_score" in parsed_row:
                        parsed_row["priority_score"] = _parse_csv_number(
                            parsed_row["priority_score"]
                        )
                    if "estimated_travel_minutes" in parsed_row:
                        parsed_row["estimated_travel_minutes"] = _parse_csv_number(
                            parsed_row["estimated_travel_minutes"]
                        )
                    zones_list.append(parsed_row)
        else:
            raise AllocationInputError(
                f"Unsupported file extension '{suffix}' for '{zones_path}': use .json or .csv."
            )
    except AllocationInputError:
        raise
    except Exception as exc:
        raise AllocationInputError(f"Failed to read '{zones_path}': {exc}") from exc

    results = allocate_resources(zones_list, team_count)

    if output_path is not None:
        out_p = Path(output_path)
        if not out_p.is_absolute() and not out_p.parent.exists():
            out_p = Path(__file__).resolve().parent / out_p
        out_suffix = out_p.suffix.lower()

        if out_suffix == ".json":
            with out_p.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        elif out_suffix == ".csv":
            fieldnames = [
                "team_id",
                "assigned_zone_id",
                "priority_score",
                "assignment_status",
                "reason",
                "available_team_count",
                "rank",
                "source",
            ]
            with out_p.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
        else:
            raise AllocationInputError(
                f"Unsupported output extension '{out_suffix}' for '{output_path}': use .json or .csv."
            )

    return results


def generate_sample_files() -> None:
    """Generate the 6 mandatory ILLUSTRATIVE sample files in the module folder."""
    base_dir = Path(__file__).resolve().parent

    # 1. sample_zones.json (4 zones: ZONE_B 84.5, ZONE_C 71.0, ZONE_D 58.0, ZONE_A 14.0)
    sample_zones = [
        {
            "zone_id": "ZONE_B",
            "priority_score": 84.5,
            "estimated_travel_minutes": 12.0,
            "reason": "Dense neighbourhood with cut-off road",
            "data_label": "ILLUSTRATIVE",
        },
        {
            "zone_id": "ZONE_C",
            "priority_score": 71.0,
            "estimated_travel_minutes": 18.0,
            "reason": "Village near district hospital",
            "data_label": "ILLUSTRATIVE",
        },
        {
            "zone_id": "ZONE_D",
            "priority_score": 58.0,
            "estimated_travel_minutes": 25.0,
            "reason": "Low-lying riverfront area",
            "data_label": "ILLUSTRATIVE",
        },
        {
            "zone_id": "ZONE_A",
            "priority_score": 14.0,
            "estimated_travel_minutes": 8.0,
            "reason": "Empty agricultural field",
            "data_label": "ILLUSTRATIVE",
        },
    ]

    sample_zones_json_path = base_dir / "sample_zones.json"
    with sample_zones_json_path.open("w", encoding="utf-8") as f:
        json.dump(sample_zones, f, indent=2)

    # 2. sample_zones.csv
    sample_zones_csv_path = base_dir / "sample_zones.csv"
    with sample_zones_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "zone_id",
                "priority_score",
                "estimated_travel_minutes",
                "reason",
                "data_label",
            ],
        )
        writer.writeheader()
        for z in sample_zones:
            writer.writerow(z)

    # 3. sample_zones_ties.json (Equal-score zones testing travel-time and alphabetical tie-breaks)
    sample_ties = [
        {
            "zone_id": "ZONE_C",
            "priority_score": 75.0,
            "estimated_travel_minutes": 20.0,
            "data_label": "ILLUSTRATIVE",
        },
        {
            "zone_id": "ZONE_B",
            "priority_score": 75.0,
            "estimated_travel_minutes": 10.0,
            "data_label": "ILLUSTRATIVE",
        },
        {
            "zone_id": "ZONE_E",
            "priority_score": 60.0,
            "data_label": "ILLUSTRATIVE",
        },
        {
            "zone_id": "ZONE_D",
            "priority_score": 60.0,
            "data_label": "ILLUSTRATIVE",
        },
    ]
    with (base_dir / "sample_zones_ties.json").open("w", encoding="utf-8") as f:
        json.dump(sample_ties, f, indent=2)

    # 4, 5, 6. Generate sample_output_3teams.json, sample_output_2teams.json, sample_output_0teams.json
    allocate_from_file(
        str(sample_zones_json_path), 3, str(base_dir / "sample_output_3teams.json")
    )
    allocate_from_file(
        str(sample_zones_json_path), 2, str(base_dir / "sample_output_2teams.json")
    )
    allocate_from_file(
        str(sample_zones_json_path), 0, str(base_dir / "sample_output_0teams.json")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloodSense AI M8 Resource Allocator CLI")
    parser.add_argument("--input", type=str, default="sample_zones.json", help="Input zones JSON/CSV")
    parser.add_argument("--teams", type=int, default=3, help="Number of available response teams")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON/CSV path")
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate all 6 required sample files in resource_allocator/",
    )
    args = parser.parse_args()

    if args.generate_samples:
        generate_sample_files()
        print("Generated all 6 sample files successfully.")
    else:
        rows = allocate_from_file(args.input, args.teams, args.output)
        print(json.dumps({"allocations": rows, "summary": summarize(rows)}, indent=2))