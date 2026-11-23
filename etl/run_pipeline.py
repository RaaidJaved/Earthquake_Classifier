"""ETL Pipeline Orchestrator.

Executes the complete extraction, validation, cleaning, target label engineering,
and database loading pipeline for seismic telemetry data.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from dotenv import load_dotenv

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from etl.extract.usgs_client import USGSClient
from etl.validate.schema import validate_raw_records
from etl.transform.cleaner import clean_and_transform_earthquakes, compute_summary_stats
from etl.load.db_loader import get_db_engine, load_dataframe_to_db

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ETLPipeline")


def run_etl(feed: str = "all_week", database_url: str | None = None) -> dict:
    """Runs the end-to-end Earthquake ETL pipeline."""
    load_dotenv()
    logger.info("=" * 60)
    logger.info(f"Starting Earthquake ETL Pipeline (Feed: {feed})")
    logger.info("=" * 60)

    # 1. EXTRACT
    logger.info("Step 1/4: Extracting seismic telemetry from USGS API...")
    client = USGSClient()
    raw_features = client.fetch_feed(feed=feed)
    logger.info(f"Extracted {len(raw_features)} raw features")

    # 2. VALIDATE
    logger.info("Step 2/4: Validating schemas with Pydantic...")
    validated_records, rejected_count = validate_raw_records(raw_features)
    logger.info(f"Validation complete: {len(validated_records)} valid, {rejected_count} rejected")

    # 3. TRANSFORM & FEATURE ENGINEERING
    logger.info("Step 3/4: Cleaning telemetry & engineering target label (mag >= 5.0)...")
    cleaned_df = clean_and_transform_earthquakes(validated_records)
    stats = compute_summary_stats(cleaned_df)
    logger.info(f"Summary: {stats}")

    # 4. LOAD
    logger.info("Step 4/4: Loading records into Cloud SQL Database...")
    engine = get_db_engine(database_url)
    loaded_count = load_dataframe_to_db(cleaned_df, engine=engine)
    logger.info(f"Loaded {loaded_count} records into database")

    logger.info("=" * 60)
    logger.info("ETL Pipeline Finished Successfully!")
    logger.info("=" * 60)

    return {
        "status": "success",
        "extracted_count": len(raw_features),
        "validated_count": len(validated_records),
        "loaded_count": loaded_count,
        "summary": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Earthquake ETL Pipeline")
    parser.add_argument(
        "--feed",
        type=str,
        default=os.getenv("USGS_FEED", "all_week"),
        help="USGS feed name: all_day, all_week, all_month, 2.5_month",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Optional override for DATABASE_URL",
    )
    args = parser.parse_args()

    try:
        run_etl(feed=args.feed, database_url=args.db_url)
    except Exception as e:
        logger.error(f"ETL Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
