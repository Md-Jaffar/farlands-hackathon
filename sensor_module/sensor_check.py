from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


class SensorInputError(ValueError):
    """
    Raised ONLY for whole-file or config problems:
    file not found, missing required columns (timestamp, sensor_id, water_level),
    unreadable config, or 0 data rows.
    """
    pass


DEFAULT_CONFIG: dict[str, Any] = {
    "timestamp_format": "%Y-%m-%d %H:%M:%S",
    "expected_interval_minutes": 15,
    "gap_factor": 1.5,
    "stale_minutes": 60,
    "water_level_min_m": 0.0,
    "water_level_max_m": 15.0,
    "rainfall_min_mm": 0.0,
    "rainfall_max_mm": 200.0,
    "max_rate_m_per_hour": 1.5,
    "jump_recovery_tolerance_m": 0.15,
    "accept_level_after": 3,
    "frozen_tolerance": 0.001,
    "frozen_min_repeats": 6,
}

REQUIRED_COLUMNS = {"timestamp", "sensor_id", "water_level"}


def _resolve_path(path_str: str) -> Path:
    """Resolve path directly or relative to sensor_check.py directory (Rule 3)."""
    p = Path(path_str)
    if p.exists() or p.is_absolute():
        return p
    alt = Path(__file__).resolve().parent / p
    return alt if alt.exists() else p


def _load_config(config_path: str) -> dict[str, Any]:
    """Load and validate JSON configuration file."""
    if not isinstance(config_path, str) or not config_path.strip():
        raise SensorInputError("Invalid config_path: must be a non-empty path string.")
    resolved = _resolve_path(config_path)
    if not resolved.exists() or not resolved.is_file():
        raise SensorInputError(f"Config file not found: '{config_path}'.")
    try:
        with resolved.open("r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        if not isinstance(user_cfg, dict):
            raise SensorInputError(f"Config file '{config_path}' must contain a JSON object.")
        merged = dict(DEFAULT_CONFIG)
        merged.update(user_cfg)
        return merged
    except SensorInputError:
        raise
    except Exception as exc:
        raise SensorInputError(f"Unreadable config file '{config_path}': {exc}") from exc


def _parse_timestamp(ts_val: Any, ts_format: str) -> datetime | None:
    """Parse timestamp string using exact ts_format, returning None on bad format."""
    if ts_val is None:
        return None
    if isinstance(ts_val, float) and math.isnan(ts_val):
        return None
    s = str(ts_val).strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, ts_format)
    except ValueError:
        return None


def _parse_optional_float(val: Any) -> tuple[float | None, str]:
    """
    Parse a numeric cell.
    Returns (float_val, state) where state is 'OK', 'EMPTY', or 'INVALID'.
    """
    if val is None:
        return None, "EMPTY"
    if isinstance(val, bool):
        return None, "INVALID"
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return None, "EMPTY"
        return float(val), "OK"
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None, "EMPTY"
    try:
        f = float(s)
        if math.isnan(f) or math.isinf(f):
            return None, "INVALID"
        return f, "OK"
    except ValueError:
        return None, "INVALID"

def _clean_str_or_none(val: Any) -> str | None:
    """Convert optional string cell (like zone_id) to clean string or None."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    return s


def analyse_dataframe(
    df: pd.DataFrame,
    config: dict[str, Any],
    reference_time: str | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate sensor stream DataFrame and return sorted list of flagged reading dictionaries.
    """
    if not isinstance(df, pd.DataFrame):
        raise SensorInputError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(config, dict):
        raise SensorInputError("Input 'config' must be a configuration dictionary.")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        raise SensorInputError(f"Missing required column(s): {missing_str}.")

    if len(df) == 0:
        raise SensorInputError("Input sensor dataset contains 0 data rows.")

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)

    ts_format = str(cfg["timestamp_format"])
    interval_min = int(cfg["expected_interval_minutes"])
    gap_factor = float(cfg["gap_factor"])
    stale_min = float(cfg["stale_minutes"])
    wl_min = float(cfg["water_level_min_m"])
    wl_max = float(cfg["water_level_max_m"])
    rain_min = float(cfg.get("rainfall_min_mm", 0.0))
    rain_max = float(cfg.get("rainfall_max_mm", 200.0))
    max_rate = float(cfg["max_rate_m_per_hour"])
    recovery_tol = float(cfg.get("jump_recovery_tolerance_m", 0.15))
    accept_after = int(cfg.get("accept_level_after", 3))
    frozen_tol = float(cfg["frozen_tolerance"])
    frozen_min = int(cfg["frozen_min_repeats"])

    # Parse reference_time if provided
    ref_dt: datetime | None = None
    if reference_time is not None:
        ref_dt = _parse_timestamp(reference_time, ts_format)
        if ref_dt is None:
            raise SensorInputError(
                f"Invalid reference_time '{reference_time}': does not match format '{ts_format}'."
            )

    # Convert DataFrame rows into internal records without mutating input df
    raw_records: list[dict[str, Any]] = []
    valid_dts: list[datetime] = []

    for idx, row in df.iterrows():
        raw_ts = "" if row["timestamp"] is None or (isinstance(row["timestamp"], float) and math.isnan(row["timestamp"])) else str(row["timestamp"]).strip()
        sensor_id = "" if row["sensor_id"] is None or (isinstance(row["sensor_id"], float) and math.isnan(row["sensor_id"])) else str(row["sensor_id"]).strip()
        if not sensor_id:
            raise SensorInputError(f"Row {idx} has an empty or missing 'sensor_id'.")

        parsed_dt = _parse_timestamp(raw_ts, ts_format)
        if parsed_dt is not None:
            if ref_dt is not None and parsed_dt > ref_dt:
                # Ignore readings strictly after reference_time
                continue
            valid_dts.append(parsed_dt)

        zone_id = _clean_str_or_none(row["zone_id"]) if "zone_id" in df.columns else None
        raw_wl = row["water_level"]
        raw_rain = row["rainfall"] if "rainfall" in df.columns else None

        raw_records.append(
            {
                "orig_idx": idx,
                "raw_ts": raw_ts,
                "dt": parsed_dt,
                "sensor_id": sensor_id,
                "zone_id": zone_id,
                "raw_wl": raw_wl,
                "raw_rain": raw_rain,
            }
        )

    if not raw_records:
        return []

    if ref_dt is None:
        ref_dt = max(valid_dts) if valid_dts else None

    # Group records by sensor_id
    by_sensor: dict[str, list[dict[str, Any]]] = {}
    for rec in raw_records:
        by_sensor.setdefault(rec["sensor_id"], []).append(rec)

    output_rows: list[dict[str, Any]] = []

    for sensor_id in sorted(by_sensor.keys()):
        sensor_recs = by_sensor[sensor_id]

        # Separate valid-timestamp rows from bad-timestamp rows, sorting valid ones chronologically
        valid_recs = [r for r in sensor_recs if r["dt"] is not None]
        bad_ts_recs = [r for r in sensor_recs if r["dt"] is None]
        valid_recs.sort(key=lambda r: (r["dt"], r["orig_idx"]))

        # Known zone_id for synthetic rows (MISSING_ROW / STALE_SENSOR)
        sensor_zone = next((r["zone_id"] for r in sensor_recs if r["zone_id"] is not None), None)

        seen_timestamps: set[datetime] = set()
        prev_valid_dt: datetime | None = None

        # State tracking for SUDDEN_JUMP and FROZEN
        prev_ok_level: float | None = None
        prev_ok_dt: datetime | None = None
        candidate_jump_level: float | None = None
        candidate_jump_dt: datetime | None = None
        candidate_streak: int = 0

        frozen_anchor_val: float | None = None
        frozen_count: int = 0

        for rec in valid_recs:
            cur_dt: datetime = rec["dt"]
            cur_ts_str = cur_dt.strftime(ts_format)
            zone_id = rec["zone_id"] or sensor_zone

            wl_val, wl_state = _parse_optional_float(rec["raw_wl"])
            rain_val, rain_state = _parse_optional_float(rec["raw_rain"])

            # 1. Check DUPLICATE_TIMESTAMP first
            if cur_dt in seen_timestamps:
                output_rows.append(
                    {
                        "timestamp": cur_ts_str,
                        "sensor_id": sensor_id,
                        "zone_id": zone_id,
                        "water_level": wl_val,
                        "rainfall": rain_val,
                        "status": "SUSPECT",
                        "reason": f"Duplicate timestamp {cur_ts_str} for sensor {sensor_id}",
                        "is_usable": False,
                        "rule": "DUPLICATE_TIMESTAMP",
                    }
                )
                continue

            # 2. Check MISSING_ROW gap before processing current valid timestamp
            if prev_valid_dt is not None:
                gap_minutes = (cur_dt - prev_valid_dt).total_seconds() / 60.0
                if gap_minutes > gap_factor * interval_min:
                    expected_dt = prev_valid_dt + timedelta(minutes=interval_min)
                    while expected_dt < cur_dt:
                        miss_ts_str = expected_dt.strftime(ts_format)
                        output_rows.append(
                            {
                                "timestamp": miss_ts_str,
                                "sensor_id": sensor_id,
                                "zone_id": sensor_zone,
                                "water_level": None,
                                "rainfall": None,
                                "status": "MISSING",
                                "reason": (
                                    f"Missing expected reading at {miss_ts_str} "
                                    f"(gap of {gap_minutes:.0f} min exceeds "
                                    f"{gap_factor * interval_min:.1f} min threshold)"
                                ),
                                "is_usable": False,
                                "rule": "MISSING_ROW",
                            }
                        )
                        expected_dt += timedelta(minutes=interval_min)

            seen_timestamps.add(cur_dt)
            prev_valid_dt = cur_dt

            # 3. Check MISSING_VALUE / INVALID_VALUE on water_level
            if wl_state == "EMPTY":
                output_rows.append(
                    {
                        "timestamp": cur_ts_str,
                        "sensor_id": sensor_id,
                        "zone_id": zone_id,
                        "water_level": None,
                        "rainfall": rain_val,
                        "status": "MISSING",
                        "reason": "Missing water_level value (empty cell)",
                        "is_usable": False,
                        "rule": "MISSING_VALUE",
                    }
                )
                continue

            if wl_state == "INVALID" or wl_val is None:
                raw_text = str(rec["raw_wl"]).strip()
                output_rows.append(
                    {
                        "timestamp": cur_ts_str,
                        "sensor_id": sensor_id,
                        "zone_id": zone_id,
                        "water_level": None,
                        "rainfall": rain_val,
                        "status": "MISSING",
                        "reason": f"Invalid non-numeric water_level value '{raw_text}'",
                        "is_usable": False,
                        "rule": "INVALID_VALUE",
                    }
                )
                continue

            # 4. Check OUT_OF_RANGE on water_level and rainfall
            if wl_val < wl_min:
                output_rows.append(
                    {
                        "timestamp": cur_ts_str,
                        "sensor_id": sensor_id,
                        "zone_id": zone_id,
                        "water_level": wl_val,
                        "rainfall": rain_val,
                        "status": "SUSPECT",
                        "reason": (
                            f"Out of range: water_level {wl_val} m is below minimum bound "
                            f"water_level_min_m ({wl_min} m)"
                        ),
                        "is_usable": False,
                        "rule": "OUT_OF_RANGE",
                    }
                )
                continue

            if wl_val > wl_max:
                output_rows.append(
                    {
                        "timestamp": cur_ts_str,
                        "sensor_id": sensor_id,
                        "zone_id": zone_id,
                        "water_level": wl_val,
                        "rainfall": rain_val,
                        "status": "SUSPECT",
                        "reason": (
                            f"Out of range: water_level {wl_val} m exceeds maximum bound "
                            f"water_level_max_m ({wl_max} m)"
                        ),
                        "is_usable": False,
                        "rule": "OUT_OF_RANGE",
                    }
                )
                continue

            if rain_val is not None and rain_val < rain_min:
                output_rows.append(
                    {
                        "timestamp": cur_ts_str,
                        "sensor_id": sensor_id,
                        "zone_id": zone_id,
                        "water_level": wl_val,
                        "rainfall": rain_val,
                        "status": "SUSPECT",
                        "reason": (
                            f"Out of range: rainfall {rain_val} mm is below minimum bound "
                            f"rainfall_min_mm ({rain_min} mm)"
                        ),
                        "is_usable": False,
                        "rule": "OUT_OF_RANGE",
                    }
                )
                continue

            if rain_val is not None and rain_val > rain_max:
                output_rows.append(
                    {
                        "timestamp": cur_ts_str,
                        "sensor_id": sensor_id,
                        "zone_id": zone_id,
                        "water_level": wl_val,
                        "rainfall": rain_val,
                        "status": "SUSPECT",
                        "reason": (
                            f"Out of range: rainfall {rain_val} mm exceeds maximum bound "
                            f"rainfall_max_mm ({rain_max} mm)"
                        ),
                        "is_usable": False,
                        "rule": "OUT_OF_RANGE",
                    }
                )
                continue

            # Update FROZEN counter on valid in-range water_level readings
            if frozen_anchor_val is not None and abs(wl_val - frozen_anchor_val) <= frozen_tol:
                frozen_count += 1
            else:
                frozen_anchor_val = wl_val
                frozen_count = 1

            # 5. Check SUDDEN_JUMP & recovery / sustained rise stabilization
            is_jump = False
            jump_reason = ""

            if prev_ok_level is not None and prev_ok_dt is not None:
                hours_elapsed = (cur_dt - prev_ok_dt).total_seconds() / 3600.0
                minutes_elapsed = (cur_dt - prev_ok_dt).total_seconds() / 60.0
                delta = wl_val - prev_ok_level
                rate = abs(delta) / hours_elapsed if hours_elapsed > 0 else 0.0

                # Check if we are in a post-jump candidate sequence
                if candidate_streak > 0 and candidate_jump_level is not None and candidate_jump_dt is not None:
                    # Case A: Reading returns within recovery_tol (0.15m) of pre-jump OK level -> immediate OK
                    if abs(wl_val - prev_ok_level) <= recovery_tol:
                        candidate_streak = 0
                        candidate_jump_level = None
                        candidate_jump_dt = None
                    else:
                        # Case B: Check if reading is stable relative to the jumped level
                        step_hours = (cur_dt - candidate_jump_dt).total_seconds() / 3600.0
                        step_rate = (
                            abs(wl_val - candidate_jump_level) / step_hours
                            if step_hours > 0
                            else 0.0
                        )
                        if step_rate <= max_rate:
                            candidate_streak += 1
                            candidate_jump_level = wl_val
                            candidate_jump_dt = cur_dt
                            if candidate_streak >= accept_after:
                                # Sustained genuine rise confirmed after accept_level_after stable readings!
                                prev_ok_level = wl_val
                                prev_ok_dt = cur_dt
                                candidate_streak = 0
                                candidate_jump_level = None
                                candidate_jump_dt = None
                            else:
                                is_jump = True
                                sign = "+" if delta >= 0 else ""
                                jump_reason = (
                                    f"Sudden jump: {sign}{delta:.2f} m in {minutes_elapsed:.0f} min "
                                    f"({rate:.2f} m/hour) exceeds the {max_rate} m/hour limit "
                                    f"(stabilizing {candidate_streak}/{accept_after})"
                                )
                        else:
                            # Jumped again wildly
                            candidate_streak = 1
                            candidate_jump_level = wl_val
                            candidate_jump_dt = cur_dt
                            is_jump = True
                            sign = "+" if delta >= 0 else ""
                            jump_reason = (
                                f"Sudden jump: {sign}{delta:.2f} m in {minutes_elapsed:.0f} min "
                                f"({rate:.2f} m/hour) exceeds the {max_rate} m/hour limit"
                            )
                else:
                    # Normal state: compare directly against prev_ok_level
                    if hours_elapsed > 0 and rate > max_rate:
                        is_jump = True
                        candidate_streak = 1
                        candidate_jump_level = wl_val
                        candidate_jump_dt = cur_dt
                        sign = "+" if delta >= 0 else ""
                        jump_reason = (
                            f"Sudden jump: {sign}{delta:.2f} m in {minutes_elapsed:.0f} min "
                            f"({rate:.2f} m/hour) exceeds the {max_rate} m/hour limit"
                        )

            if is_jump:
                output_rows.append(
                    {
                        "timestamp": cur_ts_str,
                        "sensor_id": sensor_id,
                        "zone_id": zone_id,
                        "water_level": wl_val,
                        "rainfall": rain_val,
                        "status": "SUSPECT",
                        "reason": jump_reason,
                        "is_usable": False,
                        "rule": "SUDDEN_JUMP",
                    }
                )
                continue

            # 6. Check FROZEN (only on water_level, never on rainfall)
            if frozen_count >= frozen_min:
                output_rows.append(
                    {
                        "timestamp": cur_ts_str,
                        "sensor_id": sensor_id,
                        "zone_id": zone_id,
                        "water_level": wl_val,
                        "rainfall": rain_val,
                        "status": "SUSPECT",
                        "reason": (
                            f"Frozen sensor: water_level repeated at {wl_val} m for "
                            f"{frozen_count} consecutive readings (limit {frozen_min})"
                        ),
                        "is_usable": False,
                        "rule": "FROZEN",
                    }
                )
                prev_ok_level = wl_val
                prev_ok_dt = cur_dt
                continue

            # 7. Reading is OK!
            prev_ok_level = wl_val
            prev_ok_dt = cur_dt
            output_rows.append(
                {
                    "timestamp": cur_ts_str,
                    "sensor_id": sensor_id,
                    "zone_id": zone_id,
                    "water_level": wl_val,
                    "rainfall": rain_val,
                    "status": "OK",
                    "reason": f"Normal reading: water_level {wl_val} m within expected bounds and rate limits",
                    "is_usable": True,
                    "rule": "NONE",
                }
            )

        # 8. Handle BAD_TIMESTAMP rows for this sensor
        for rec in bad_ts_recs:
            wl_val, _ = _parse_optional_float(rec["raw_wl"])
            rain_val, _ = _parse_optional_float(rec["raw_rain"])
            output_rows.append(
                {
                    "timestamp": rec["raw_ts"],
                    "sensor_id": sensor_id,
                    "zone_id": rec["zone_id"] or sensor_zone,
                    "water_level": wl_val,
                    "rainfall": rain_val,
                    "status": "SUSPECT",
                    "reason": (
                        f"Unparseable timestamp '{rec['raw_ts']}': does not match "
                        f"expected format '{ts_format}'"
                    ),
                    "is_usable": False,
                    "rule": "BAD_TIMESTAMP",
                }
            )

        # 9. Check STALE_SENSOR against ref_dt
        if ref_dt is not None and prev_valid_dt is not None:
            silent_minutes = (ref_dt - prev_valid_dt).total_seconds() / 60.0
            if silent_minutes > stale_min:
                ref_ts_str = ref_dt.strftime(ts_format)
                last_ts_str = prev_valid_dt.strftime(ts_format)
                output_rows.append(
                    {
                        "timestamp": ref_ts_str,
                        "sensor_id": sensor_id,
                        "zone_id": sensor_zone,
                        "water_level": None,
                        "rainfall": None,
                        "status": "STALE",
                        "reason": (
                            f"Stale sensor: no update since {last_ts_str} "
                            f"({silent_minutes:.0f} min elapsed exceeds {stale_min:.0f} min limit)"
                        ),
                        "is_usable": False,
                        "rule": "STALE_SENSOR",
                    }
                )

    # Sort deterministically by sensor_id ascending, then timestamp ascending
    output_rows.sort(key=lambda r: (r["sensor_id"], r["timestamp"]))
    return output_rows

def analyse_sensor_file(
    input_path: str,
    config_path: str,
    reference_time: str | None = None,
) -> list[dict[str, Any]]:
    """
    Read sensor CSV file and JSON config file, run anomaly checks, and return sorted row dicts.
    """
    if not isinstance(input_path, str) or not input_path.strip():
        raise SensorInputError("Invalid input_path: must be a non-empty file path string.")

    resolved_in = _resolve_path(input_path)
    if not resolved_in.exists() or not resolved_in.is_file():
        raise SensorInputError(f"Sensor input file not found: '{input_path}'.")

    cfg = _load_config(config_path)

    try:
        # Read keeping raw strings so empty vs invalid text ('ERR') can be distinguished accurately
        df = pd.read_csv(resolved_in, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise SensorInputError(f"Failed to read CSV file '{input_path}': {exc}") from exc

    return analyse_dataframe(df, cfg, reference_time=reference_time)


def write_output_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write flagged output dictionaries to CSV."""
    fieldnames = [
        "timestamp",
        "sensor_id",
        "zone_id",
        "water_level",
        "rainfall",
        "status",
        "reason",
        "is_usable",
        "rule",
    ]
    out_p = Path(output_path)
    with out_p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def generate_sample_and_scenario_files() -> None:
    """
    Generate sample_input.csv, sample_output.csv, injection_log.csv, and scenario_input.csv
    synced with WP01's Assam flood timeline (2016-08-11 12:00:00 to 2016-08-12 12:00:00).
    """
    base_dir = Path(__file__).resolve().parent
    cfg_path = base_dir / "sensor_config.json"
    if not cfg_path.exists():
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)

    fieldnames = [
        "timestamp",
        "sensor_id",
        "zone_id",
        "latitude",
        "longitude",
        "water_level",
        "rainfall",
        "data_label",
    ]

    # -------------------------------------------------------------------------
    # PART 1: sample_input.csv & injection_log.csv (All 8 fault patterns + normal)
    # -------------------------------------------------------------------------
    sample_rows: list[dict[str, Any]] = []
    injection_rows: list[dict[str, str]] = []
    t0 = datetime(2016, 8, 11, 12, 0, 0)

    # S_TEST_1: Normal baseline + OUT_OF_RANGE (t+15m) + SUDDEN_JUMP spike (t+45m) + MISSING_VALUE (t+75m) + INVALID_VALUE (t+90m)
    s1_schedule = [
        (0, "2.10", "1.0", None),
        (15, "18.50", "1.2", "OUT_OF_RANGE"),       # > 15.0m max
        (30, "2.15", "1.5", None),
        (45, "5.55", "1.5", "SUDDEN_JUMP"),         # +3.4m in 15m spike
        (60, "2.18", "1.0", None),                  # recovers within 0.15m of 2.15
        (75, "", "0.5", "MISSING_VALUE"),           # empty water_level
        (90, "ERR", "0.5", "INVALID_VALUE"),        # non-numeric water_level
        (105, "2.22", "0.0", None),
        (120, "2.25", "0.0", None),
    ]
    for mins, wl, rain, fault in s1_schedule:
        ts_str = (t0 + timedelta(minutes=mins)).strftime("%Y-%m-%d %H:%M:%S")
        sample_rows.append(
            {
                "timestamp": ts_str,
                "sensor_id": "S_TEST_1",
                "zone_id": "ZONE_A",
                "latitude": "26.1200",
                "longitude": "93.1500",
                "water_level": wl,
                "rainfall": rain,
                "data_label": "SIMULATED",
            }
        )
        if fault:
            injection_rows.append(
                {
                    "sensor_id": "S_TEST_1",
                    "start_timestamp": ts_str,
                    "end_timestamp": ts_str,
                    "injected_type": fault,
                }
            )

    # Add DUPLICATE_TIMESTAMP & BAD_TIMESTAMP on S_TEST_1
    dup_ts = (t0 + timedelta(minutes=120)).strftime("%Y-%m-%d %H:%M:%S")
    sample_rows.append(
        {
            "timestamp": dup_ts,
            "sensor_id": "S_TEST_1",
            "zone_id": "ZONE_A",
            "latitude": "26.1200",
            "longitude": "93.1500",
            "water_level": "2.25",
            "rainfall": "0.0",
            "data_label": "SIMULATED",
        }
    )
    injection_rows.append(
        {
            "sensor_id": "S_TEST_1",
            "start_timestamp": dup_ts,
            "end_timestamp": dup_ts,
            "injected_type": "DUPLICATE_TIMESTAMP",
        }
    )

    sample_rows.append(
        {
            "timestamp": "2016/08/11 99:99:99",
            "sensor_id": "S_TEST_1",
            "zone_id": "ZONE_A",
            "latitude": "26.1200",
            "longitude": "93.1500",
            "water_level": "2.26",
            "rainfall": "0.0",
            "data_label": "SIMULATED",
        }
    )
    injection_rows.append(
        {
            "sensor_id": "S_TEST_1",
            "start_timestamp": "2016/08/11 99:99:99",
            "end_timestamp": "2016/08/11 99:99:99",
            "injected_type": "BAD_TIMESTAMP",
        }
    )

    # S_TEST_2: FROZEN run (6 identical readings at 3.00m) + MISSING_ROW gap (skips 105m)
    for step_idx, mins in enumerate([0, 15, 30, 45, 60, 75, 90, 120]):
        ts_str = (t0 + timedelta(minutes=mins)).strftime("%Y-%m-%d %H:%M:%S")
        wl = "3.00" if step_idx < 6 else ("3.05" if step_idx == 6 else "3.10")
        sample_rows.append(
            {
                "timestamp": ts_str,
                "sensor_id": "S_TEST_2",
                "zone_id": "ZONE_C",
                "latitude": "26.1400",
                "longitude": "93.2000",
                "water_level": wl,
                "rainfall": "0.0",
                "data_label": "SIMULATED",
            }
        )
    frozen_ts = (t0 + timedelta(minutes=75)).strftime("%Y-%m-%d %H:%M:%S")
    injection_rows.append(
        {
            "sensor_id": "S_TEST_2",
            "start_timestamp": frozen_ts,
            "end_timestamp": frozen_ts,
            "injected_type": "FROZEN",
        }
    )
    missed_ts = (t0 + timedelta(minutes=105)).strftime("%Y-%m-%d %H:%M:%S")
    injection_rows.append(
        {
            "sensor_id": "S_TEST_2",
            "start_timestamp": missed_ts,
            "end_timestamp": missed_ts,
            "injected_type": "MISSING_ROW",
        }
    )

    # S_TEST_3: STALE_SENSOR (stops at t+30m while file ends at t+120m -> 90m silent > 60m)
    for mins in [0, 15, 30]:
        ts_str = (t0 + timedelta(minutes=mins)).strftime("%Y-%m-%d %H:%M:%S")
        sample_rows.append(
            {
                "timestamp": ts_str,
                "sensor_id": "S_TEST_3",
                "zone_id": "ZONE_D",
                "latitude": "26.0800",
                "longitude": "93.1000",
                "water_level": "1.50",
                "rainfall": "0.0",
                "data_label": "SIMULATED",
            }
        )
    ref_end_ts = (t0 + timedelta(minutes=120)).strftime("%Y-%m-%d %H:%M:%S")
    injection_rows.append(
        {
            "sensor_id": "S_TEST_3",
            "start_timestamp": ref_end_ts,
            "end_timestamp": ref_end_ts,
            "injected_type": "STALE_SENSOR",
        }
    )

    sample_in_path = base_dir / "sample_input.csv"
    with sample_in_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sample_rows:
            writer.writerow(r)

    inj_path = base_dir / "injection_log.csv"
    with inj_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sensor_id", "start_timestamp", "end_timestamp", "injected_type"]
        )
        writer.writeheader()
        for r in injection_rows:
            writer.writerow(r)

    sample_out_rows = analyse_sensor_file(str(sample_in_path), str(cfg_path))
    write_output_csv(sample_out_rows, base_dir / "sample_output.csv")

    # -------------------------------------------------------------------------
    # PART 2: scenario_input.csv (6-Sensor Story for Assam Aug 11-12, 2016)
    # 97 steps of 15-min intervals from 2016-08-11 12:00:00 to 2016-08-12 12:00:00
    # -------------------------------------------------------------------------
    scenario_rows: list[dict[str, Any]] = []
    num_steps = 97  # 24 hours * 4 + 1

    for i in range(num_steps):
        cur_dt = t0 + timedelta(minutes=15 * i)
        ts_str = cur_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Rainfall profile: rises ahead of flood peak around step 40-65
        if 20 <= i < 40:
            rain_b = round(4.0 + (i - 20) * 0.6, 2)
        elif 40 <= i <= 65:
            rain_b = round(16.0 - abs(i - 50) * 0.4, 2)
        else:
            rain_b = 1.0

        # SENSOR_01, SENSOR_02, SENSOR_03 in ZONE_B: Gradual genuine rise (<= 0.12m per 15m = 0.48m/hr <= 1.5m/hr)
        for s_idx, (s_id, base_lvl, lat, lon) in enumerate(
            [
                ("SENSOR_01", 2.50, "26.1510", "93.2110"),
                ("SENSOR_02", 2.65, "26.1540", "93.2150"),
                ("SENSOR_03", 2.40, "26.1490", "93.2080"),
            ]
        ):
            if i < 24:
                rise = i * 0.02
            elif i < 72:
                rise = 24 * 0.02 + (i - 24) * 0.08
            else:
                rise = 24 * 0.02 + 48 * 0.08 + (i - 72) * 0.01
            wl_b = round(base_lvl + rise + s_idx * 0.003, 3)
            scenario_rows.append(
                {
                    "timestamp": ts_str,
                    "sensor_id": s_id,
                    "zone_id": "ZONE_B",
                    "latitude": lat,
                    "longitude": lon,
                    "water_level": f"{wl_b:.3f}",
                    "rainfall": f"{rain_b:.2f}",
                    "data_label": "SIMULATED",
                }
            )

        # SENSOR_04 in ZONE_C: Smaller rise + 6 identical consecutive readings at steps 50..55 (FROZEN)
        if 50 <= i <= 55:
            wl_c = 3.120
        elif i < 50:
            wl_c = round(1.90 + i * 0.024, 3)
        else:
            wl_c = round(3.120 + (i - 55) * 0.015, 3)
        scenario_rows.append(
            {
                "timestamp": ts_str,
                "sensor_id": "SENSOR_04",
                "zone_id": "ZONE_C",
                "latitude": "26.1120",
                "longitude": "93.1850",
                "water_level": f"{wl_c:.3f}",
                "rainfall": f"{max(0.0, rain_b * 0.7):.2f}",
                "data_label": "SIMULATED",
            }
        )

        # SENSOR_05 in ZONE_A: Only sensor in ZONE_A; flat baseline with a single 1-reading spike at step 48
        if i == 48:
            wl_a = 4.900  # Spike (+3.40m in 15 min) -> SUDDEN_JUMP
        else:
            wl_a = round(1.500 + (i % 3) * 0.01, 3)
        scenario_rows.append(
            {
                "timestamp": ts_str,
                "sensor_id": "SENSOR_05",
                "zone_id": "ZONE_A",
                "latitude": "26.0500",
                "longitude": "92.9500",
                "water_level": f"{wl_a:.3f}",
                "rainfall": "0.00",
                "data_label": "SIMULATED",
            }
        )

        # SENSOR_06 in ZONE_D: Flat and calm, stops reporting after step 88 (10:00:00, which is 120 min before 12:00:00)
        if i <= 88:
            wl_d = round(1.800 + (i % 4) * 0.01, 3)
            scenario_rows.append(
                {
                    "timestamp": ts_str,
                    "sensor_id": "SENSOR_06",
                    "zone_id": "ZONE_D",
                    "latitude": "26.0900",
                    "longitude": "93.3100",
                    "water_level": f"{wl_d:.3f}",
                    "rainfall": "0.50",
                    "data_label": "SIMULATED",
                }
            )

    scenario_in_path = base_dir / "scenario_input.csv"
    with scenario_in_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in scenario_rows:
            writer.writerow(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloodSense AI M2 Sensor Trust Layer CLI")
    parser.add_argument("--input", type=str, default="sample_input.csv", help="Input sensor CSV")
    parser.add_argument(
        "--config", type=str, default="sensor_config.json", help="Sensor config JSON"
    )
    parser.add_argument("--ref-time", type=str, default=None, help="Optional reference_time")
    parser.add_argument("--output", type=str, default=None, help="Optional output CSV path")
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate sample_input.csv, sample_output.csv, injection_log.csv, and scenario_input.csv",
    )
    args = parser.parse_args()

    if args.generate_samples:
        generate_sample_and_scenario_files()
        print("Generated sample_input.csv, sample_output.csv, injection_log.csv, and scenario_input.csv.")
    else:
        results = analyse_sensor_file(args.input, args.config, reference_time=args.ref_time)
        if args.output:
            write_output_csv(results, args.output)
        print(json.dumps(results[:10], indent=2))
        print(f"Total processed rows: {len(results)}")