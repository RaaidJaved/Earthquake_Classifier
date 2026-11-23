"""Cloud SQL (PostgreSQL - Neon / Supabase) Database Loader Module.

Handles schema initialization, idempotent batch loading, and connection pooling.
"""

from __future__ import annotations
import logging
import os
from typing import Optional
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

load_dotenv()

logger = logging.getLogger(__name__)

metadata = MetaData()

# Define the earthquakes SQL table
earthquakes_table = Table(
    "earthquakes",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("magnitude", Float, nullable=False),
    Column("is_significant", Integer, nullable=False, index=True),
    Column("place", String(255)),
    Column("event_time", DateTime(timezone=True), index=True),
    Column("event_time_iso", String(64)),
    Column("time_epoch", BigInteger),
    Column("longitude", Float, nullable=False),
    Column("latitude", Float, nullable=False),
    Column("depth", Float, nullable=False),
    Column("nst", Integer, nullable=True),
    Column("gap", Float, nullable=True),
    Column("dmin", Float, nullable=True),
    Column("rms", Float, nullable=True),
    Column("mag_type", String(32)),
    Column("alert", String(32)),
    Column("felt", BigInteger, nullable=True),
    Column("tsunami", Integer, default=0),
    Column("sig", Integer),
    Column("net", String(32)),
    Column("status", String(32)),
    Column("class_label", String(64)),
    Column("nst_missing", Integer, default=0),
    Column("gap_missing", Integer, default=0),
    Column("dmin_missing", Integer, default=0),
    Column("rms_missing", Integer, default=0),
)


def normalize_database_url(url: Optional[str]) -> str:
    """Ensures database connection string is compatible with SQLAlchemy and PostgreSQL drivers."""
    if not url:
        return "sqlite:///./earthquakes.db"
    
    # Fix Heroku/Supabase legacy postgres:// prefix to postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)

    return url


def get_db_engine(database_url: Optional[str] = None) -> Engine:
    """Creates a configured SQLAlchemy Engine for Cloud SQL (Neon/Supabase) or local fallback."""
    raw_url = database_url or os.getenv("DATABASE_URL", "sqlite:///./earthquakes.db")
    db_uri = normalize_database_url(raw_url)

    engine_kwargs = {"echo": False}
    if "sqlite" in db_uri:
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # PostgreSQL pool configuration for cloud environments (Neon / Supabase)
        engine_kwargs["pool_pre_ping"] = True
        engine_kwargs["pool_recycle"] = 300

    logger.info(f"Connecting to database dialect: {db_uri.split(':')[0]}")
    return create_engine(db_uri, **engine_kwargs)


def init_db_schema(engine: Optional[Engine] = None) -> None:
    """Creates tables if they do not exist, and migrates column types if needed."""
    eng = engine or get_db_engine()
    metadata.create_all(eng)
    
    # In PostgreSQL, ensure time_epoch is BIGINT even if previously created as INT
    if eng.dialect.name == "postgresql":
        try:
            with eng.begin() as conn:
                conn.execute(text("ALTER TABLE earthquakes ALTER COLUMN time_epoch TYPE BIGINT;"))
                conn.execute(text("ALTER TABLE earthquakes ALTER COLUMN felt TYPE BIGINT;"))
        except Exception as e:
            logger.debug(f"Column migration check: {e}")

    logger.info("Database schema initialized successfully (table 'earthquakes')")


def load_dataframe_to_db(df: pd.DataFrame, engine: Optional[Engine] = None) -> int:
    """Loads cleaned earthquake DataFrame into PostgreSQL or SQLite.
    
    Implements idempotent upsert/insert to prevent duplicates on repeated ETL runs.
    
    Returns:
        Number of rows inserted/updated.
    """
    if df.empty:
        logger.warning("DataFrame is empty, nothing to load into database")
        return 0

    eng = engine or get_db_engine()
    init_db_schema(eng)

    is_postgres = eng.dialect.name == "postgresql"

    # Convert DataFrame to records for bulk loading
    target_columns = [c.name for c in earthquakes_table.columns]
    load_df = df[[c for c in target_columns if c in df.columns]].copy()

    # Convert DataFrame to clean dictionary records with native Python types
    raw_records = load_df.to_dict(orient="records")
    records = []
    for row in raw_records:
        clean_row = {}
        for k, v in row.items():
            if pd.isna(v):
                clean_row[k] = None
            elif hasattr(v, "to_pydatetime"):
                clean_row[k] = v.to_pydatetime() if is_postgres else (v.isoformat() if hasattr(v, "isoformat") else str(v))
            elif isinstance(v, (int, float)) and pd.notnull(v):
                # Ensure numpy integer types are converted to standard Python int/float
                clean_row[k] = int(v) if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and k in ("time_epoch", "nst", "tsunami", "sig", "felt", "is_significant")) else v
            else:
                clean_row[k] = v
        records.append(clean_row)

    chunk_size = 300
    with eng.begin() as conn:
        if is_postgres:
            # PostgreSQL ON CONFLICT (id) DO UPDATE (upsert) in chunks
            for i in range(0, len(records), chunk_size):
                chunk = records[i : i + chunk_size]
                stmt = pg_insert(earthquakes_table).values(chunk)
                update_dict = {
                    col.name: stmt.excluded[col.name]
                    for col in earthquakes_table.columns
                    if col.name != "id"
                }
                upsert_stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_=update_dict
                )
                conn.execute(upsert_stmt)
        else:
            # SQLite fallback: INSERT OR REPLACE
            for i in range(0, len(records), chunk_size):
                chunk = records[i : i + chunk_size]
                keys = list(chunk[0].keys())
                columns_str = ", ".join(keys)
                placeholders = ", ".join([f":{k}" for k in keys])
                sql = text(f"INSERT OR REPLACE INTO earthquakes ({columns_str}) VALUES ({placeholders})")
                conn.execute(sql, chunk)

    logger.info(f"Successfully loaded {len(records)} earthquake telemetry records into database")
    return len(records)


# Alias for backward-compatibility
EarthquakeRecordDB = earthquakes_table
