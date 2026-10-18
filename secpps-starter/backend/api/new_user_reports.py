
import csv
import io
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.auth import get_current_user
from app_services.report_pdf import PdfReport
from database import get_db
from models_auth import User
from models_profile import ForecastJob

router = APIRouter()

HORIZONS = (
    ("24h", "forecast: next 24 hours (hourly)", "Next 24 hours"),
    ("7day", "forecast: next 7 days (daily)", "Next 7 days"),
    ("1month", "forecast: next 30 days (daily)", "Next 30 days"),
)


def _load(db: Session, user: User):
    job = db.query(ForecastJob).filter(ForecastJob.user_id == user.id).first()
    if job is None or job.status != "done":
        raise HTTPException(
            status_code=404,
            detail="No finished forecast yet. Upload your data and wait for the forecast to finish.",
        )
    return job, json.loads(job.history_json), json.loads(job.result_json)


def _usable(result: dict, key: str):
    data = (result.get("horizons") or {}).get(key)
    return data if data and "error" not in data else None


@router.get("/forecast/new-user/report.csv")
def report_csv(
    rate_per_kwh: Optional[float] = Query(None, ge=0, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _, history, result = _load(db, current_user)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["section", "period_start", "kwh"] + (["cost"] if rate_per_kwh else []))

    def add(section, period, kwh):
        writer.writerow([section, period, kwh] + ([round(kwh * rate_per_kwh, 4)] if rate_per_kwh else []))

    for point in history:
        add("history (hourly)", point["timestamp"], point["kwh"])
    for key, label, _ in HORIZONS:
        data = _usable(result, key)
        if data:
            for p in data["predictions"]:
                add(label, p.get("timestamp") or p.get("date"), p["predicted_kwh"])

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=energy_report_uploaded_data.csv"},
    )


@router.get("/forecast/new-user/report.pdf")
def report_pdf(
    rate_per_kwh: Optional[float] = Query(None, ge=0, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job, history, result = _load(db, current_user)
    info = json.loads(job.info_json or "{}")
    place = (result.get("location") or {}).get("resolved_name") or job.location

    values = [p["kwh"] for p in history]
    total = sum(values)
    days = len(values) / 24
    peak = max(history, key=lambda p: p["kwh"])

    pdf = PdfReport("SECPPS Energy Report (uploaded data)")
    pdf.line(f"Location: {place} ({job.country_code}{', ' + job.state if job.state else ''})")
    pdf.line(f"Data used: {info.get('start', history[0]['timestamp'])[:10]} to "
             f"{info.get('end', history[-1]['timestamp'])[:10]} ({days:.0f} days)")
    pdf.line(f"Generated: {datetime.utcnow():%d %b %Y %H:%M} UTC")

    pdf.heading("Your usage (history)")
    rows = [
        ["Total consumption", f"{total:,.1f} kWh"],
        ["Average per day", f"{total / days:,.2f} kWh"],
        ["Peak hour", f"{peak['kwh']:.2f} kWh on {peak['timestamp'][:16].replace('T', ' ')}"],
    ]
    if rate_per_kwh:
        rows.append(["Estimated cost (history)", f"${total * rate_per_kwh:,.2f} at ${rate_per_kwh:.2f}/kWh"])
    pdf.table(["Measure", "Value"], rows, [180, 300])

    pdf.heading("Forecast")
    header = ["Horizon", "Total", "Average per day"] + (["Est. cost"] if rate_per_kwh else [])
    widths = [130, 110, 130] + ([100] if rate_per_kwh else [])
    rows, methods = [], []
    for key, _, label in HORIZONS:
        data = _usable(result, key)
        if not data:
            rows.append([label, "not available", ""] + ([""] if rate_per_kwh else []))
            continue
        n_days = 1 if key == "24h" else len(data["predictions"])
        row = [label, f"{data['total_kwh']:,.1f} kWh", f"{data['total_kwh'] / n_days:,.1f} kWh"]
        if rate_per_kwh:
            row.append(f"${data['total_kwh'] * rate_per_kwh:,.2f}")
        rows.append(row)
        methods.append(f"{label}: {data['method']}")
    pdf.table(header, rows, widths)
    pdf.gap(6)
    for text in methods:
        pdf.line(text, size=9, indent=6)

    week = _usable(result, "7day")
    if week:
        pdf.heading("Next 7 days")
        rows = []
        for p in week["predictions"]:
            row = [p["date"], f"{p['predicted_kwh']:.2f} kWh"]
            if rate_per_kwh:
                row.append(f"${p['predicted_kwh'] * rate_per_kwh:,.2f}")
            rows.append(row)
        pdf.table(["Date", "Forecast"] + (["Est. cost"] if rate_per_kwh else []), rows, [130, 130, 100])

    pdf.heading("About this forecast")
    for text in result.get("limitations", []):
        pdf.line(f"- {text}", size=9.5, indent=4)

    pdf.heading("Cost")
    if rate_per_kwh:
        pdf.line(f"Cost figures use the rate you entered (${rate_per_kwh:.4f} per kWh), not a tariff "
                 f"from your retailer.")
    else:
        pdf.line("No cost is shown. Enter your electricity rate ($ per kWh) when downloading the "
                 "report to include an estimate.")

    return StreamingResponse(
        io.BytesIO(pdf.finish()),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=energy_report_uploaded_data.pdf"},
    )