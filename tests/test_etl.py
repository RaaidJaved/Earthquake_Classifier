"""Comprehensive Test Suite for Earthquake Classifier Pipeline."""

import unittest
import pandas as pd
from pydantic import ValidationError
from etl.validate.schema import (
    USGSGeometry,
    USGSProperties,
    USGSEarthquakeRaw,
    ValidatedEarthquakeRecord,
    validate_raw_records,
)
from etl.transform.cleaner import clean_and_transform_earthquakes, compute_summary_stats
from etl.load.db_loader import get_db_engine, init_db_schema, load_dataframe_to_db
from app.services.eda_service import EDAService


class TestEarthquakePipeline(unittest.TestCase):

    def test_pydantic_geometry_validation(self):
        """Test coordinates validation rules."""
        # Valid coordinates: [lon, lat, depth]
        geom = USGSGeometry(coordinates=[-122.4, 37.7, 10.0])
        self.assertEqual(geom.coordinates[0], -122.4)
        self.assertEqual(geom.coordinates[1], 37.7)

        # Invalid latitude (> 90)
        with self.assertRaises(ValidationError):
            USGSGeometry(coordinates=[-122.4, 999.0, 10.0])

        # Invalid longitude (< -180)
        with self.assertRaises(ValidationError):
            USGSGeometry(coordinates=[-200.0, 37.7, 10.0])

    def test_pydantic_raw_to_validated_feature(self):
        """Test full raw GeoJSON parsing and validation."""
        raw_feature = {
            "type": "Feature",
            "id": "test_event_001",
            "properties": {
                "mag": 5.4,
                "place": "10 km SW of Anchorage, Alaska",
                "time": 1700000000000,
                "nst": 42,
                "gap": 65.0,
                "dmin": 0.12,
                "rms": 0.25,
                "magType": "mw",
                "alert": "green",
                "tsunami": 0,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [-149.9, 61.2, 35.0],
            },
        }

        validated_list, rejected = validate_raw_records([raw_feature])
        self.assertEqual(len(validated_list), 1)
        self.assertEqual(rejected, 0)

        record = validated_list[0]
        self.assertEqual(record.id, "test_event_001")
        self.assertEqual(record.magnitude, 5.4)
        self.assertEqual(record.longitude, -149.9)
        self.assertEqual(record.latitude, 61.2)
        self.assertEqual(record.depth, 35.0)

    def test_cleaner_and_target_engineering(self):
        """Test Pandas transformation and target label engineering (mag >= 5.0)."""
        raw_features = [
            # Significant quake (mag 5.2 >= 5.0) -> is_significant = 1
            {
                "type": "Feature",
                "id": "quake_sig",
                "properties": {"mag": 5.2, "place": "Fiji Islands", "time": 1700000000000, "nst": 30},
                "geometry": {"type": "Point", "coordinates": [-178.0, -18.0, 500.0]},
            },
            # Minor quake (mag 2.1 < 5.0) -> is_significant = 0
            {
                "type": "Feature",
                "id": "quake_minor",
                "properties": {"mag": 2.1, "place": "Southern California", "time": 1700000005000, "nst": None},
                "geometry": {"type": "Point", "coordinates": [-118.0, 34.0, 10.0]},
            },
            # Quake with null magnitude -> should be dropped
            {
                "type": "Feature",
                "id": "quake_no_mag",
                "properties": {"mag": None, "place": "Unknown", "time": 1700000010000},
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0, 0.0]},
            },
        ]

        records, _ = validate_raw_records(raw_features)
        df = clean_and_transform_earthquakes(records)

        # 1 null magnitude dropped -> 2 remaining
        self.assertEqual(len(df), 2)

        # Check target labels
        sig_row = df[df["id"] == "quake_sig"].iloc[0]
        minor_row = df[df["id"] == "quake_minor"].iloc[0]

        self.assertEqual(sig_row["is_significant"], 1)
        self.assertEqual(minor_row["is_significant"], 0)

        # Check missingness indicators
        self.assertEqual(minor_row["nst_missing"], 1)
        self.assertEqual(sig_row["nst_missing"], 0)

        # Summary stats
        stats = compute_summary_stats(df)
        self.assertEqual(stats["total_records"], 2)
        self.assertEqual(stats["significant_count"], 1)
        self.assertEqual(stats["minor_count"], 1)

    def test_db_loading_and_eda_service(self):
        """Test loading into database and EDA query aggregations."""
        test_db_url = "sqlite:///:memory:"
        engine = get_db_engine(test_db_url)
        init_db_schema(engine)

        raw_features = [
            {
                "type": "Feature",
                "id": "eq_1",
                "properties": {"mag": 6.0, "place": "Tokyo, Japan", "time": 1700000000000, "nst": 50, "gap": 40.0, "dmin": 0.1, "rms": 0.15, "alert": "yellow"},
                "geometry": {"type": "Point", "coordinates": [139.7, 35.6, 40.0]},
            },
            {
                "type": "Feature",
                "id": "eq_2",
                "properties": {"mag": 1.8, "place": "Nevada", "time": 1700000005000, "nst": 12, "gap": 80.0, "dmin": 0.2, "rms": 0.1},
                "geometry": {"type": "Point", "coordinates": [-115.0, 36.0, 5.0]},
            },
            {
                "type": "Feature",
                "id": "eq_3",
                "properties": {"mag": 2.4, "place": "Hawaii", "time": 1700000010000, "nst": None, "gap": None, "dmin": None, "rms": None},
                "geometry": {"type": "Point", "coordinates": [-155.0, 19.5, 8.0]},
            },
        ]

        records, _ = validate_raw_records(raw_features)
        df = clean_and_transform_earthquakes(records)
        loaded = load_dataframe_to_db(df, engine=engine)
        self.assertEqual(loaded, 3)

        # Idempotent re-load test (should not crash or duplicate)
        reloaded = load_dataframe_to_db(df, engine=engine)
        self.assertEqual(reloaded, 3)

        # Test EDA service queries
        eda = EDAService(engine=engine)
        summary = eda.get_summary()
        self.assertEqual(summary["total_records"], 3)
        self.assertEqual(summary["significant_count"], 1)
        self.assertEqual(summary["minor_count"], 2)

        class_balance = eda.get_class_balance()
        self.assertEqual(class_balance["counts"], [2, 1])

        missingness = eda.get_missingness_analysis()
        self.assertEqual(len(missingness["attributes"]), 6)
        # eq_3 has null nst, so overall missingness should be > 0%
        self.assertTrue(any(val > 0.0 for val in missingness["overall"]))


if __name__ == "__main__":
    unittest.main()

