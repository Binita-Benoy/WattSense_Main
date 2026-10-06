"""
Onboarding profile + new-user forecast job.

Both live under the same Base as User (database.Base), so the foreign keys to
users.id resolve. Tables are created automatically on startup.
"""
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func

from database import Base


class UserProfile(Base):
    """Which route the user chose after logging in: "new" or "existing"."""
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    user_type = Column(String(20), nullable=False)  # "new" | "existing"
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ForecastJob(Base):
    """
    A new user's uploaded history and the forecast built from it. One row per
    user; uploading again replaces it. The result is stored here so reloading the
    dashboard or switching horizon never re-runs the (slow) model.
    """
    __tablename__ = "forecast_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    status = Column(String(20), nullable=False, default="queued")  # queued|running|done|failed
    progress = Column(String(120))
    error = Column(Text)
    location = Column(String(255))
    country_code = Column(String(2))
    state = Column(String(10))
    warnings_json = Column(Text)
    info_json = Column(Text)
    history_json = Column(Text)  # cleaned hourly history: [{"timestamp", "kwh"}, ...]
    result_json = Column(Text)   # output of run_live_forecast_all_horizons
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))