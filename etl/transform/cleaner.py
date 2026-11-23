"""Data Cleaning and Feature Engineering Module.

Uses Pandas to clean raw seismic telemetry, handle missing sensor data,
and engineer target labels for earthquake classification.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List
import pandas as pd
from etl.validate.schema import ValidatedEarthquakeRecord

logger = logging.getLogger(__name__)

SIGNIFICANT_MAGNITUDE_THRESHOLD = 5.0


def clean_and_transform_earthquakes(records: List[ValidatedEarthquakeRecord]) -> pd.DataFrame:
    """Transforms validated records into a clean, ML-ready Pandas DataFrame.

    Engineering steps:
    1. Filter out records missing ground-truth magnitude.
    2. Engineer binary classification target `is_significant` (magnitude >= 5.0).
    3. Convert epoch timestamp to standard UTC datetime.
    4. Handle sensor missingness and add explicit missingness flags.
    5. Deduplicate based on unique earthquake identifier.
    """
    if not records:
        logger.warning("Empty records list provided to cleaner")
        return pd.DataFrame()

    # Convert Pydantic records to DataFrame
    data_dicts = [r.model_dump() for r in records]
    df = pd.DataFrame(data_dicts)

    initial_count = len(df)

    # 1. Deduplicate by USGS event ID
    df = df.drop_duplicates(subset=["id"], keep="last").copy()

    # 2. Filter records where magnitude is missing
    df = df.dropna(subset=["magnitude"]).copy()
    valid_mag_count = len(df)
    dropped_mags = initial_count - valid_mag_count
    if dropped_mags > 0:
        logger.info(f"Dropped {dropped_mags} events with null magnitude")

    # 3. Target label engineering
    # Classification target: 1 = Significant (mag >= 5.0), 0 = Minor (< 5.0)
    df["is_significant"] = (df["magnitude"] >= SIGNIFICANT_MAGNITUDE_THRESHOLD).astype(int)

    # Human-readable class name for EDA
    df["class_label"] = df["is_significant"].map({1: "Significant (>= 5.0)", 0: "Minor (< 5.0)"})

    # 4. Datetime transformation
    if "time_epoch" in df.columns:
        df["event_time"] = pd.to_datetime(df["time_epoch"], unit="ms", utc=True)
        df["event_time_iso"] = df["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    # 5. Missingness Flags (for sensor quality missingness analysis)
    # Sensor quality metrics: nst (stations), gap (coverage gap), dmin (distance), rms (timing residual)
    sensor_cols = ["nst", "gap", "dmin", "rms"]
    for col in sensor_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
        else:
            df[col] = None
            df[f"{col}_missing"] = 1

    # 6. Spatial coordinate sanitization
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce").fillna(0.0)

    # 7. Station quality imputation / feature tags
    # Keep original values with nulls for missingness EDA, but create clean features for ML
    df["place"] = df["place"].fillna("Unknown Location")
    df["mag_type"] = df["mag_type"].fillna("unknown")

    logger.info(
        f"Cleaned {len(df)} records. "
        f"Class balance: {df['is_significant'].value_counts().to_dict()}"
    )

    return df


def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates analytical summary statistics from the transformed DataFrame."""
    if df.empty:
        return {
            "total_records": 0,
            "significant_count": 0,
            "minor_count": 0,
            "significant_ratio_pct": 0.0,
            "min_magnitude": 0.0,
            "max_magnitude": 0.0,
            "mean_magnitude": 0.0,
        }

    total = len(df)
    sig_count = int(df["is_significant"].sum())
    minor_count = total - sig_count
    ratio = (sig_count / total * 100.0) if total > 0 else 0.0

    return {
        "total_records": total,
        "significant_count": sig_count,
        "minor_count": minor_count,
        "significant_ratio_pct": round(ratio, 2),
        "min_magnitude": round(float(df["magnitude"].min()), 2),
        "max_magnitude": round(float(df["magnitude"].max()), 2),
        "mean_magnitude": round(float(df["magnitude"].mean()), 2),
        "avg_depth": round(float(df["depth"].mean()), 2),
    }
