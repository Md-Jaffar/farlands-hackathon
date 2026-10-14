from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from sensor_check import (
    DEFAULT_CONFIG,
    SensorInputError,
    analyse_dataframe,
    analyse_sensor_file,
)


@pytest.fixture
def cfg() -> dict:
    return dict(DEFAULT_CONFIG)


# Test 1: Normal gradual rise stays OK when within bounds and <= 1.5 m/hour rate limit
def test_01_normal_gradual_rise(cfg: dict) -> None:
    df = pd.DataFrame(
        [
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S1", "water_level": 2.00, "rainfall": 1.0},
            {"timestamp": "2016-08-11 12:15:00", "sensor_id": "S1", "water_level": 2.30, "rainfall": 2.0},
            {"timestamp": "2016-08-11 12:30:00", "sensor_id": "S1", "water_level": 2.60, "rainfall": 2.5},
        ]
    )
    res = analyse_dataframe(df, cfg)
    assert len(res) == 3
    assert all(r["status"] == "OK" and r["rule"] == "NONE" and r["is_usable"] is True for r in res)


# Test 2: Out-of-range value flagged SUSPECT (OUT_OF_RANGE) and names breached bound
def test_02_out_of_range_bounds(cfg: dict) -> None:
    df = pd.DataFrame(
        [
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S1", "water_level": -0.5, "rainfall": 0.0},
            {"timestamp": "2016-08-11 12:15:00", "sensor_id": "S1", "water_level": 16.2, "rainfall": 0.0},
            {"timestamp": "2016-08-11 12:30:00", "sensor_id": "S1", "water_level": 2.0, "rainfall": -1.0},
        ]
    )
    res = analyse_dataframe(df, cfg)
    assert [r["rule"] for r in res] == ["OUT_OF_RANGE", "OUT_OF_RANGE", "OUT_OF_RANGE"]
    assert "water_level_min_m" in res[0]["reason"]
    assert "water_level_max_m" in res[1]["reason"]
    assert "rainfall_min_mm" in res[2]["reason"]


# Test 3: Sudden spike returning to normal -> spike is SUSPECT, next reading within 0.15m is OK
def test_03_sudden_spike_recovery(cfg: dict) -> None:
    df = pd.DataFrame(
        [
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S1", "water_level": 2.00},
            {"timestamp": "2016-08-11 12:15:00", "sensor_id": "S1", "water_level": 5.40},  # +3.4m spike
            {"timestamp": "2016-08-11 12:30:00", "sensor_id": "S1", "water_level": 2.10},  # within 0.15m of 2.00
        ]
    )
    res = analyse_dataframe(df, cfg)
    assert res[0]["status"] == "OK"
    assert res[1]["status"] == "SUSPECT" and res[1]["rule"] == "SUDDEN_JUMP"
    assert res[2]["status"] == "OK" and res[2]["rule"] == "NONE"


# Test 4: Sustained genuine rise -> initial jump SUSPECT, becomes OK after accept_level_after (3) readings
def test_04_sustained_genuine_rise(cfg: dict) -> None:
    df = pd.DataFrame(
        [
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S1", "water_level": 2.00},
            {"timestamp": "2016-08-11 12:15:00", "sensor_id": "S1", "water_level": 4.50},  # jump #1 (SUSPECT)
            {"timestamp": "2016-08-11 12:30:00", "sensor_id": "S1", "water_level": 4.55},  # stable #2 (SUSPECT)
            {"timestamp": "2016-08-11 12:45:00", "sensor_id": "S1", "water_level": 4.60},  # stable #3 -> accepted OK!
            {"timestamp": "2016-08-11 13:00:00", "sensor_id": "S1", "water_level": 4.65},  # continues OK
        ]
    )
    res = analyse_dataframe(df, cfg)
    assert [r["status"] for r in res] == ["OK", "SUSPECT", "SUSPECT", "OK", "OK"]
    assert res[1]["rule"] == "SUDDEN_JUMP"
    assert res[3]["rule"] == "NONE"


# Test 5: Frozen reading flagged at 6 repeats; repeated 0.0 rainfall never triggers FROZEN
def test_05_frozen_water_level_and_zero_rainfall_ignored(cfg: dict) -> None:
    rows = []
    for i in range(7):
        rows.append(
            {
                "timestamp": f"2016-08-11 12:{i*15:02d}:00" if i < 4 else f"2016-08-11 13:{(i-4)*15:02d}:00",
                "sensor_id": "S_FROZ",
                "water_level": 3.00,
                "rainfall": 0.0,
            }
        )
        rows.append(
            {
                "timestamp": f"2016-08-11 12:{i*15:02d}:00" if i < 4 else f"2016-08-11 13:{(i-4)*15:02d}:00",
                "sensor_id": "S_DRY_RAIN",
                "water_level": 2.00 + i * 0.05,
                "rainfall": 0.0,
            }
        )
    df = pd.DataFrame(rows)
    res = analyse_dataframe(df, cfg)
    froz_rows = [r for r in res if r["sensor_id"] == "S_FROZ"]
    dry_rows = [r for r in res if r["sensor_id"] == "S_DRY_RAIN"]

    assert [r["rule"] for r in froz_rows] == ["NONE"] * 5 + ["FROZEN", "FROZEN"]
    assert all(r["status"] == "OK" for r in dry_rows)


# Test 6: Missing value, invalid value, and missing row gap
def test_06_missing_invalid_and_missing_row_gap(cfg: dict) -> None:
    df = pd.DataFrame(
        [
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S1", "water_level": "2.00"},
            {"timestamp": "2016-08-11 12:15:00", "sensor_id": "S1", "water_level": ""},
            {"timestamp": "2016-08-11 12:30:00", "sensor_id": "S1", "water_level": "ERR"},
            # Skips 12:45:00 -> triggers MISSING_ROW
            {"timestamp": "2016-08-11 13:00:00", "sensor_id": "S1", "water_level": "2.10"},
        ]
    )
    res = analyse_dataframe(df, cfg)
    assert len(res) == 5
    assert [r["rule"] for r in res] == [
        "NONE",
        "MISSING_VALUE",
        "INVALID_VALUE",
        "MISSING_ROW",
        "NONE",
    ]
    assert res[3]["timestamp"] == "2016-08-11 12:45:00"
    assert res[3]["water_level"] is None


# Test 7: Sensor stops updating -> emits exactly one STALE_SENSOR row at reference_time
def test_07_stale_sensor_emitted_once(cfg: dict) -> None:
    df = pd.DataFrame(
        [
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S_STALE", "water_level": 1.5},
            {"timestamp": "2016-08-11 12:15:00", "sensor_id": "S_STALE", "water_level": 1.5},
            {"timestamp": "2016-08-11 14:00:00", "sensor_id": "S_ACTIVE", "water_level": 2.0},
        ]
    )
    res = analyse_dataframe(df, cfg)
    stale_rows = [r for r in res if r["rule"] == "STALE_SENSOR"]
    assert len(stale_rows) == 1
    assert stale_rows[0]["sensor_id"] == "S_STALE"
    assert stale_rows[0]["timestamp"] == "2016-08-11 14:00:00"
    assert stale_rows[0]["status"] == "STALE"


# Test 8: reference_time filtering ignores future rows and updates staleness dynamically
def test_08_reference_time_filtering(cfg: dict) -> None:
    df = pd.DataFrame(
        [
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S1", "water_level": 1.5},
            {"timestamp": "2016-08-11 12:15:00", "sensor_id": "S1", "water_level": 1.6},
            {"timestamp": "2016-08-11 14:00:00", "sensor_id": "S1", "water_level": 9.9},
        ]
    )
    res_early = analyse_dataframe(df, cfg, reference_time="2016-08-11 12:15:00")
    assert len(res_early) == 2
    assert all(r["status"] == "OK" for r in res_early)

    res_late = analyse_dataframe(df, cfg, reference_time="2016-08-11 13:30:00")
    assert any(r["rule"] == "STALE_SENSOR" for r in res_late)


# Test 9: Duplicate & bad timestamps flagged SUSPECT without crashing
def test_09_duplicate_and_bad_timestamps(cfg: dict) -> None:
    df = pd.DataFrame(
        [
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S1", "water_level": 2.0},
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S1", "water_level": 2.1},
            {"timestamp": "NOT_A_TIMESTAMP", "sensor_id": "S1", "water_level": 2.2},
        ]
    )
    res = analyse_dataframe(df, cfg)
    rules = [r["rule"] for r in res]
    assert "DUPLICATE_TIMESTAMP" in rules
    assert "BAD_TIMESTAMP" in rules


# Test 10: All-normal file produces 100% OK rows with zero false flags
def test_10_all_normal_sensors_zero_false_flags(cfg: dict) -> None:
    base_dir = Path(__file__).resolve().parent
    res = analyse_sensor_file(str(base_dir / "scenario_input.csv"), str(base_dir / "sensor_config.json"))
    zone_b_rows = [r for r in res if r["sensor_id"] in ("SENSOR_01", "SENSOR_02", "SENSOR_03")]
    assert len(zone_b_rows) == 97 * 3
    assert all(r["status"] == "OK" and r["rule"] == "NONE" for r in zone_b_rows)


# Test 11: Multiple sensors evaluated independently
def test_11_multiple_sensors_independent_state(cfg: dict) -> None:
    df = pd.DataFrame(
        [
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S_GOOD", "water_level": 2.0},
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S_BAD", "water_level": 25.0},
            {"timestamp": "2016-08-11 12:15:00", "sensor_id": "S_GOOD", "water_level": 2.1},
            {"timestamp": "2016-08-11 12:15:00", "sensor_id": "S_BAD", "water_level": 2.1},
        ]
    )
    res = analyse_dataframe(df, cfg)
    good = [r for r in res if r["sensor_id"] == "S_GOOD"]
    bad = [r for r in res if r["sensor_id"] == "S_BAD"]
    assert all(r["status"] == "OK" for r in good)
    assert bad[0]["rule"] == "OUT_OF_RANGE"
    assert bad[1]["status"] == "OK"


# Test 12: Unsorted input produces exact same sorted output as pre-sorted input
def test_12_unsorted_input_deterministic(cfg: dict) -> None:
    df_sorted = pd.DataFrame(
        [
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S1", "water_level": 2.0},
            {"timestamp": "2016-08-11 12:15:00", "sensor_id": "S1", "water_level": 2.1},
            {"timestamp": "2016-08-11 12:00:00", "sensor_id": "S2", "water_level": 3.0},
            {"timestamp": "2016-08-11 12:15:00", "sensor_id": "S2", "water_level": 3.1},
        ]
    )
    df_shuffled = df_sorted.iloc[[3, 1, 2, 0]].reset_index(drop=True)
    assert analyse_dataframe(df_sorted, cfg) == analyse_dataframe(df_shuffled, cfg)


# Test 13: Missing required column or missing file raises SensorInputError
def test_13_missing_column_or_file_raises_error(cfg: dict) -> None:
    bad_df = pd.DataFrame([{"timestamp": "2016-08-11 12:00:00", "sensor_id": "S1"}])
    with pytest.raises(SensorInputError, match="water_level"):
        analyse_dataframe(bad_df, cfg)

    with pytest.raises(SensorInputError):
        analyse_sensor_file("non_existent_file_12345.csv", "sensor_config.json")


# Test 14: Original non-null water_level and rainfall floats preserved unchanged
def test_14_original_float_values_preserved(cfg: dict) -> None:
    df = pd.DataFrame(
        [
            {
                "timestamp": "2016-08-11 12:00:00",
                "sensor_id": "S1",
                "water_level": 2.345678,
                "rainfall": 12.87654,
            }
        ]
    )
    res = analyse_dataframe(df, cfg)
    assert res[0]["water_level"] == 2.345678
    assert res[0]["rainfall"] == 12.87654


# Test 15: Detection against injection_log.csv (100% recall on injected faults, 0 false flags)
def test_15_detection_against_injection_log() -> None:
    base_dir = Path(__file__).resolve().parent
    res = analyse_sensor_file(
        str(base_dir / "sample_input.csv"), str(base_dir / "sensor_config.json")
    )
    with (base_dir / "injection_log.csv").open("r", encoding="utf-8") as f:
        injections = list(csv.DictReader(f))

    # 1. Verify every single injected anomaly in injection_log.csv is caught with the exact rule
    caught_keys: set[tuple[str, str, str]] = set()
    for inj in injections:
        s_id = inj["sensor_id"]
        ts = inj["start_timestamp"]
        expected_rule = inj["injected_type"]
        matching = [
            r
            for r in res
            if r["sensor_id"] == s_id
            and r["timestamp"] == ts
            and r["rule"] == expected_rule
        ]
        assert len(matching) >= 1, f"Missed injected anomaly: {inj}"
        caught_keys.add((s_id, ts, expected_rule))

    # 2. Verify zero normal readings were falsely flagged outside injection_log.csv
    non_ok_rows = [r for r in res if r["status"] != "OK"]
    for r in non_ok_rows:
        key = (r["sensor_id"], r["timestamp"], r["rule"])
        assert key in caught_keys, f"False positive flag on normal reading: {r}"