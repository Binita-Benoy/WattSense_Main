"""
Makes a new user's uploaded history visible to the anomaly service.

The anomaly service reads only from the database (households + energy_readings). A new user's CSV is saved in
forecast_jobs as JSON, so before checking it for anomalies it is copied into those tables once, under a household
id that belongs to that user alone:

    household_id = UPLOAD_HOUSEHOLD_BASE + user.id        (source = 'upload')

The ids start far above the dataset's (about 8 to 11 million), so uploads can't collide with dataset
households, and registration is told to ignore them (see api/auth.py). The team's synthetic demo household
(99999999) sits above them too, but that would only collide at user id 9,999,999.

sync_upload_household() is idempotent: if the stored rows already match the upload nothing is written; if the user
uploaded a different file, the old rows and their weather are replaced.
"""
import threading
from datetime import datetime

from sqlalchemy import column, delete, func, insert, select, table, text

UPLOAD_HOUSEHOLD_BASE = 90_000_000

_energy = table("energy_readings", column("household_id"), column("reading_timestamp"),
                column("consumption_kwh"), column("source"))
_weather = table("weather_readings", column("household_id"))
_sync_lock = threading.Lock()


def upload_household_id(user_id: int) -> int:
    return UPLOAD_HOUSEHOLD_BASE + int(user_id)


def _matches(db, hid: int, history: list) -> bool:
    count, first, last, total = db.execute(
        select(func.count(), func.min(_energy.c.reading_timestamp), func.max(_energy.c.reading_timestamp),
               func.coalesce(func.sum(_energy.c.consumption_kwh), 0.0)).where(_energy.c.household_id == hid)
    ).one()
    if count != len(history):
        return False
    return (
        first == datetime.fromisoformat(history[0]["timestamp"])
        and last == datetime.fromisoformat(history[-1]["timestamp"])
        and abs(float(total) - sum(p["kwh"] for p in history)) < 0.01
    )


def sync_upload_household(db, user_id: int, history: list) -> int:
    """Ensure this user's uploaded hourly history is stored; returns the household id to check."""
    hid = upload_household_id(user_id)
    with _sync_lock:
        if _matches(db, hid, history):
            return hid
        # Replace: weather too, so it is fetched fresh for the new date range (re-fetching over old rows would
        # leave duplicates, which the anomaly service's one-row-per-hour join rejects).
        db.execute(delete(_weather).where(_weather.c.household_id == hid))
        db.execute(delete(_energy).where(_energy.c.household_id == hid))
        db.execute(
            text("INSERT INTO households (household_id, source) VALUES (:h, 'upload') ON CONFLICT (household_id) DO NOTHING"),
            {"h": hid},
        )
        db.execute(
            insert(_energy),
            [{"household_id": hid, "reading_timestamp": datetime.fromisoformat(p["timestamp"]),
              "consumption_kwh": p["kwh"], "source": "upload"} for p in history],
        )
        db.commit()
    return hid


def delete_upload_household(db, user_id: int) -> None:
    """Remove a user's stored upload (readings, weather, and the household row)."""
    hid = upload_household_id(user_id)
    with _sync_lock:
        db.execute(delete(_weather).where(_weather.c.household_id == hid))
        db.execute(delete(_energy).where(_energy.c.household_id == hid))
        db.execute(text("DELETE FROM households WHERE household_id = :h AND source = 'upload'"), {"h": hid})
        db.commit()