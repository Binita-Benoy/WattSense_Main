"""
Report generation routes (US-19) for dataset households ("existing users").

  GET /api/my-household/report.csv   one row per reading in the chosen date range
  GET /api/my-household/report.pdf   consumption summary (and cost, see below)

COST: reliable tariff data isn't available yet (the Tariff model and
app_services/current_cost.py have a field mismatch), so cost is only included when the
user supplies their own rate with ?rate_per_kwh=0.30. The report labels it as their figure.
"""
import csv
import io
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app_services.report_pdf import PdfReport
from database import get_db
from models import EnergyReading
from models_auth import User
from api.auth import get_current_user

router = APIRouter()


def _get_readings(db: Session, household_id: int, start, end):
    q = db.query(EnergyReading).filter(EnergyReading.household_id == household_id)
    if start:
        q = q.filter(EnergyReading.reading_timestamp >= start)
    if end:
        q = q.filter(EnergyReading.reading_timestamp <= end)
    return q.order_by(EnergyReading.reading_timestamp).all()


@router.get("/my-household/report.csv")
def get_report_csv(
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    rate_per_kwh: Optional[float] = Query(None, ge=0, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = _get_readings(db, current_user.household_id, start, end)
    if not rows:
        raise HTTPException(status_code=404, detail="No readings found for this period.")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    header = ["timestamp", "consumption_kwh", "controlled_load_kwh", "gross_generation_kwh", "net_generation_kwh"]
    if rate_per_kwh:
        header.append("cost")
    writer.writerow(header)
    for r in rows:
        row = [r.reading_timestamp.isoformat(), r.consumption_kwh, r.controlled_load_kwh,
               r.gross_generation_kwh, r.net_generation_kwh]
        if rate_per_kwh:
            row.append(round((r.consumption_kwh or 0) * rate_per_kwh, 4))
        writer.writerow(row)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=energy_report_{current_user.household_id}.csv"},
    )


@router.get("/my-household/report.pdf")
def get_report_pdf(
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    rate_per_kwh: Optional[float] = Query(None, ge=0, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = _get_readings(db, current_user.household_id, start, end)
    if not rows:
        raise HTTPException(status_code=404, detail="No readings found for this period.")

    values = [r.consumption_kwh or 0 for r in rows]
    total = sum(values)
    days = len({r.reading_timestamp.date() for r in rows})
    peak_row = max(rows, key=lambda r: r.consumption_kwh or 0)

    pdf = PdfReport("SECPPS Energy Usage Report")
    pdf.line(f"Household ID: {current_user.household_id}")
    pdf.line(f"Period: {rows[0].reading_timestamp:%d %b %Y} to {rows[-1].reading_timestamp:%d %b %Y}")
    pdf.line(f"Generated: {datetime.utcnow():%d %b %Y %H:%M} UTC")

    pdf.heading("Consumption summary")
    pdf.table(["Measure", "Value"], [
        ["Readings", f"{len(rows):,}"],
        ["Days covered", f"{days}"],
        ["Total consumption", f"{total:,.2f} kWh"],
        ["Average per day", f"{total / days:,.2f} kWh"],
        ["Average per reading", f"{total / len(rows):.3f} kWh"],
        ["Peak reading", f"{max(values):.3f} kWh on {peak_row.reading_timestamp:%d %b %Y %H:%M}"],
    ], [180, 300])

    pdf.heading("Cost")
    if rate_per_kwh:
        pdf.table(["Measure", "Value"], [
            ["Rate used (your figure)", f"${rate_per_kwh:.4f} per kWh"],
            ["Estimated cost for the period", f"${total * rate_per_kwh:,.2f}"],
            ["Estimated cost per day", f"${total * rate_per_kwh / days:,.2f}"],
        ], [180, 300])
    else:
        pdf.line("No cost is shown. Enter your electricity rate ($ per kWh) when downloading the "
                 "report to include an estimate.")

    return StreamingResponse(
        io.BytesIO(pdf.finish()),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=energy_report_{current_user.household_id}.pdf"},
    )