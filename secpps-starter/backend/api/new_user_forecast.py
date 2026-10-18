
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import deque
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from api.auth import get_current_user
from app_services.consumption_csv import CsvValidationError, normalize_consumption_csv
from database import Base, SessionLocal, engine, get_db
from models_auth import User
from models_profile import ForecastJob

# models_profile must be imported BEFORE this line, or its tables aren't created.
Base.metadata.create_all(bind=engine)

log = logging.getLogger("new_user_forecast")
router = APIRouter()

RUNNER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "app_services", "forecast_runner.py")
)
TIMEOUT_SECONDS = int(os.getenv("FORECAST_TIMEOUT_SECONDS", "1800"))
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
_slot = threading.Semaphore(1)  # one model run at a time

# Lines printed by live_pipeline.py -> what we show the user while it runs
STAGES = [
    ("Geocoding", "Finding your location"),
    ("Fetching historical weather", "Fetching past weather for your dates"),
    ("Fetching live weather forecast", "Fetching the weather forecast"),
    ("Building feature vector", "Preparing your data"),
    ("Recursive forecast", "Training on your data and forecasting 24 hours + 7 days"),
    ("Direct multi-step forecast", "Training on your data and forecasting 1 month"),
]


def _now():
    return datetime.now(timezone.utc)


def _update(db: Session, job_id: int, **fields):
    job = db.get(ForecastJob, job_id)
    if job is None:
        return None
    for key, value in fields.items():
        setattr(job, key, value)
    db.commit()
    return job


def _fail(db: Session, job_id: int, message: str):
    _update(db, job_id, status="failed", error=message, progress=None, finished_at=_now())


def _run_job(job_id: int, csv_text: str, location: str, country_code: str, state: str):
    """Runs in a background thread: starts the pipeline process and records the outcome."""
    db = SessionLocal()
    tmp = None
    try:
        with _slot:
            _update(db, job_id, status="running", progress="Starting", started_at=_now())

            tmp = tempfile.mkdtemp(prefix="forecast_job_")
            csv_path = os.path.join(tmp, "input.csv")
            out_path = os.path.join(tmp, "result.json")
            with open(csv_path, "w", newline="") as f:
                f.write(csv_text)

            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",   # CPU only: the compose setup has no GPU
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "PYTHONUNBUFFERED": "1",      # so progress lines arrive as they're printed
            }
            cmd = [sys.executable, RUNNER, "--csv", csv_path, "--location", location,
                   "--country", country_code, "--state", state, "--out", out_path]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env)

            timed_out = []

            def _kill():
                timed_out.append(True)
                proc.kill()

            timer = threading.Timer(TIMEOUT_SECONDS, _kill)
            timer.start()
            tail = deque(maxlen=25)
            try:
                for line in proc.stdout:
                    line = line.rstrip()
                    tail.append(line)
                    for marker, label in STAGES:
                        if marker in line:
                            _update(db, job_id, progress=label)
                            break
                proc.wait()
            finally:
                timer.cancel()

            if timed_out:
                _fail(db, job_id, f"The forecast took longer than {TIMEOUT_SECONDS // 60} minutes and was stopped.")
            elif os.path.exists(out_path):
                with open(out_path) as f:
                    payload = json.load(f)
                if payload.get("ok"):
                    _update(db, job_id, status="done", progress="Done", error=None,
                            result_json=json.dumps(payload["result"], allow_nan=False),
                            finished_at=_now())
                else:
                    _fail(db, job_id, payload.get("error") or "The forecast failed.")
            else:
                last = " | ".join(tail)[-300:]
                _fail(db, job_id, f"The forecast process stopped unexpectedly (exit code {proc.returncode}). {last}")
    except Exception as exc:  # noqa: BLE001
        log.exception("Forecast job %s crashed", job_id)
        try:
            _fail(db, job_id, f"Unexpected error: {exc}")
        except Exception:  # noqa: BLE001
            log.exception("Could not record failure for job %s", job_id)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        db.close()


def _fail_stale_jobs():
    """Jobs left 'queued'/'running' by a previous process can never finish; mark them failed."""
    db = SessionLocal()
    try:
        stale = db.query(ForecastJob).filter(ForecastJob.status.in_(["queued", "running"])).all()
        for job in stale:
            job.status = "failed"
            job.progress = None
            job.error = "The server restarted while this forecast was running. Please upload your file again."
            job.finished_at = _now()
        if stale:
            db.commit()
    finally:
        db.close()


_fail_stale_jobs()


@router.post("/forecast/new-user", status_code=status.HTTP_202_ACCEPTED)
def start_new_user_forecast(
    file: UploadFile = File(...),
    location: str = Form(...),
    country_code: str = Form("AU"),
    state: str = Form("NSW"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    location = location.strip()
    country_code = (country_code or "AU").strip().upper()
    state = (state or "").strip().upper()
    if len(location) < 2:
        raise HTTPException(status_code=400, detail="Please enter your location, e.g. 'Sydney, Australia'.")
    if not (len(country_code) == 2 and country_code.isalpha()):
        raise HTTPException(status_code=400, detail="Country code must be 2 letters, e.g. AU.")

    existing = db.query(ForecastJob).filter(ForecastJob.user_id == current_user.id).first()
    if existing and existing.status in ("queued", "running"):
        raise HTTPException(status_code=409, detail="A forecast is already being built for your account.")

    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="The file is too large (15 MB maximum).")
    try:
        df, info, notes = normalize_consumption_csv(raw)
    except CsvValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    history = [
        {"timestamp": t.strftime("%Y-%m-%dT%H:%M:%S"), "kwh": float(v)}
        for t, v in zip(df["timestamp"], df["kwh"])
    ]
    csv_text = df.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S")

    fields = dict(
        status="queued", progress="Waiting to start", error=None,
        location=location, country_code=country_code, state=state,
        warnings_json=json.dumps(notes), info_json=json.dumps(info),
        history_json=json.dumps(history), result_json=None,
        created_at=_now(), started_at=None, finished_at=None,
    )
    if existing:
        for key, value in fields.items():
            setattr(existing, key, value)
        job = existing
    else:
        job = ForecastJob(user_id=current_user.id, **fields)
        db.add(job)
    db.commit()
    db.refresh(job)

    threading.Thread(
        target=_run_job, args=(job.id, csv_text, location, country_code, state), daemon=True
    ).start()
    return {"status": "queued", "warnings": notes, "info": info}


@router.get("/forecast/new-user")
def get_new_user_forecast(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(ForecastJob).filter(ForecastJob.user_id == current_user.id).first()
    if job is None:
        return {"status": "none"}

    response = {
        "status": job.status,
        "progress": job.progress,
        "location": job.location,
        "country_code": job.country_code,
        "state": job.state,
        "warnings": json.loads(job.warnings_json or "[]"),
        "info": json.loads(job.info_json or "{}"),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }
    if job.status == "failed":
        response["error"] = job.error
    if job.status == "done":
        response["history"] = json.loads(job.history_json)
        response["result"] = json.loads(job.result_json)
    return response


@router.delete("/forecast/new-user")
def delete_new_user_forecast(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(ForecastJob).filter(ForecastJob.user_id == current_user.id).first()
    if job is None:
        return {"status": "none"}
    if job.status in ("queued", "running"):
        raise HTTPException(status_code=409, detail="Please wait for the current forecast to finish.")
    db.delete(job)
    db.commit()
    return {"status": "none"}