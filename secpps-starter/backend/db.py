"""
Shared SQLAlchemy setup for the whole backend.
Reuses the same DATABASE_URL main.py already reads from the environment.

If a shared db.py already exists elsewhere in the team repo (check
app_services/README.md or docs/interfaces.md), delete this file and
import from theirs instead — don't run two separate engines against
the same database.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/secpps"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session per request, closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()