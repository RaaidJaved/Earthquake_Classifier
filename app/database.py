"""Database connection provider for the FastAPI web application.

Supports Cloud SQL (Neon / Supabase PostgreSQL) and local fallback.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

load_dotenv()


def get_db_engine() -> Engine:
    """Returns SQLAlchemy Engine with connection pooling suitable for cloud environments."""
    raw_url = os.getenv("DATABASE_URL", "sqlite:///./earthquakes.db")
    
    # Normalize Postgres URI for SQLAlchemy
    if raw_url.startswith("postgres://"):
        raw_url = raw_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif raw_url.startswith("postgresql://") and not raw_url.startswith("postgresql+"):
        raw_url = raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    connect_args = {}
    if "sqlite" in raw_url:
        connect_args["check_same_thread"] = False
        return create_engine(raw_url, connect_args=connect_args)
    
    # PostgreSQL settings
    return create_engine(
        raw_url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=10,
    )


engine = get_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session():
    """FastAPI Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
