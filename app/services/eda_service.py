"""EDA (Exploratory Data Analysis) Service Module.

Provides analytical queries and aggregations for:
1. Class Balance (Skewed ratio between minor tremors & significant quakes)
2. Missingness Analysis (% of missing values in sensor telemetry across classes)
3. Spatial Distribution (Global coordinate clusters for tectonic mapping)
"""

from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from app.database import get_db_engine


class EDAService:
    """Service to fetch and aggregate seismic EDA metrics from Cloud SQL."""

    def __init__(self, engine: Engine | None = None):
        self.engine = engine or get_db_engine()

    def _load_df(self, query: str, params: dict | None = None) -> pd.DataFrame:
        """Executes a SQL query and returns a pandas DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql_query(text(query), conn, params=params)

    def get_summary(self) -> Dict[str, Any]:
        """Calculates top-level dataset summary metrics."""
        try:
            df = self._load_df("SELECT magnitude, is_significant, depth, event_time_iso FROM earthquakes")
        except Exception:
            # Table might not exist yet if ETL hasn't run
            return {
                "total_records": 0,
                "significant_count": 0,
                "minor_count": 0,
                "significant_pct": 0.0,
                "imbalance_ratio": "0:0",
                "min_magnitude": 0.0,
                "max_magnitude": 0.0,
                "mean_magnitude": 0.0,
                "avg_depth_km": 0.0,
            }

        if df.empty:
            return {
                "total_records": 0,
                "significant_count": 0,
                "minor_count": 0,
                "significant_pct": 0.0,
                "imbalance_ratio": "0:0",
                "min_magnitude": 0.0,
                "max_magnitude": 0.0,
                "mean_magnitude": 0.0,
                "avg_depth_km": 0.0,
            }

        total = len(df)
        sig_count = int(df["is_significant"].sum())
        minor_count = total - sig_count
        sig_pct = (sig_count / total * 100.0) if total > 0 else 0.0
        ratio_str = f"1 : {round(minor_count / sig_count, 1)}" if sig_count > 0 else f"{minor_count}:0"

        return {
            "total_records": total,
            "significant_count": sig_count,
            "minor_count": minor_count,
            "significant_pct": round(sig_pct, 2),
            "imbalance_ratio": ratio_str,
            "min_magnitude": round(float(df["magnitude"].min()), 2),
            "max_magnitude": round(float(df["magnitude"].max()), 2),
            "mean_magnitude": round(float(df["magnitude"].mean()), 2),
            "avg_depth_km": round(float(df["depth"].mean()), 2),
        }

    def get_class_balance(self) -> Dict[str, Any]:
        """Calculates the class distribution between minor tremors and significant earthquakes."""
        try:
            df = self._load_df("SELECT is_significant, count(*) as count FROM earthquakes GROUP BY is_significant")
        except Exception:
            return {"labels": ["Minor (< 5.0)", "Significant (>= 5.0)"], "counts": [0, 0], "percentages": [0.0, 0.0]}

        counts_map = {int(row["is_significant"]): int(row["count"]) for _, row in df.iterrows()}
        minor = counts_map.get(0, 0)
        sig = counts_map.get(1, 0)
        total = minor + sig

        minor_pct = round((minor / total * 100.0), 2) if total > 0 else 0.0
        sig_pct = round((sig / total * 100.0), 2) if total > 0 else 0.0

        return {
            "labels": ["Minor Tremors (< 5.0)", "Significant Earthquakes (>= 5.0)"],
            "counts": [minor, sig],
            "percentages": [minor_pct, sig_pct],
            "total": total,
            "imbalance_multiplier": round((minor / sig), 1) if sig > 0 else 0,
        }

    def get_missingness_analysis(self) -> Dict[str, Any]:
        """Analyzes sensor attribute missingness across classes.
        
        Sensor and telemetry attributes investigated:
        - nst (recording station count)
        - gap (azimuthal network coverage gap)
        - dmin (distance to nearest station)
        - rms (travel time residual error)
        - alert (PAGER ground alert)
        - felt (shaking telemetry / citizen reports)
        """
        try:
            df = self._load_df("""
                SELECT is_significant, nst, gap, dmin, rms, alert, felt
                FROM earthquakes
            """)
        except Exception:
            return {"attributes": [], "overall": [], "minor_class": [], "significant_class": []}

        if df.empty:
            return {"attributes": [], "overall": [], "minor_class": [], "significant_class": []}

        attributes = [
            ("nst", "Station Count (nst)"),
            ("gap", "Coverage Gap (gap)"),
            ("dmin", "Nearest Dist (dmin)"),
            ("rms", "Time Residual (rms)"),
            ("alert", "PAGER Alert (alert)"),
            ("felt", "Felt Telemetry (felt)"),
        ]

        attr_keys = [a[0] for a in attributes]
        attr_labels = [a[1] for a in attributes]

        minor_df = df[df["is_significant"] == 0]
        sig_df = df[df["is_significant"] == 1]

        overall_missing: List[float] = []
        minor_missing: List[float] = []
        sig_missing: List[float] = []

        total_records = len(df)
        total_minor = len(minor_df)
        total_sig = len(sig_df)

        for col in attr_keys:
            # Overall missing %
            if col in df.columns:
                overall_pct = float((df[col].isna().sum() / total_records * 100.0)) if total_records > 0 else 0.0
                m_pct = float((minor_df[col].isna().sum() / total_minor * 100.0)) if total_minor > 0 else 0.0
                s_pct = float((sig_df[col].isna().sum() / total_sig * 100.0)) if total_sig > 0 else 0.0
            else:
                overall_pct = 0.0
                m_pct = 0.0
                s_pct = 0.0

            overall_missing.append(round(overall_pct, 1))
            minor_missing.append(round(m_pct, 1))
            sig_missing.append(round(s_pct, 1))

        return {
            "attributes": attr_labels,
            "overall": overall_missing,
            "minor_class": minor_missing,
            "significant_class": sig_missing,
        }

    def get_spatial_distribution(self, limit: int = 3000) -> List[Dict[str, Any]]:
        """Returns geospatial event coordinates for mapping tectonic boundary clusters."""
        try:
            df = self._load_df(f"""
                SELECT id, latitude, longitude, depth, magnitude, is_significant, place, event_time_iso, nst, gap
                FROM earthquakes
                ORDER BY event_time DESC
                LIMIT {limit}
            """)
        except Exception:
            return []

        if df.empty:
            return []

        # Return list of dictionaries for Leaflet map markers
        records = df.to_dict(orient="records")
        return records

    def get_recent_earthquakes(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent earthquake records for the telemetry preview table."""
        try:
            df = self._load_df(f"""
                SELECT id, place, magnitude, depth, is_significant, nst, gap, dmin, rms, event_time_iso
                FROM earthquakes
                ORDER BY event_time DESC
                LIMIT {limit}
            """)
            return df.where(pd.notnull(df), None).to_dict(orient="records")
        except Exception:
            return []
