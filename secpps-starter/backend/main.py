"""
SECPPS backend entrypoint.

This is intentionally a thin stub so the whole team can run `docker compose up`
on day one and see a working service, before any real logic exists.

Jazim: wire real routers in here (api/, app_services/) as they're built.
"""
import os
from fastapi import FastAPI
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/secpps")

app = FastAPI(title="SECPPS API", version="0.1.0")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/db")
def health_db():
    """Confirms the backend can actually reach Postgres/TimescaleDB."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        row = result.fetchone()
    return {"status": "ok", "db_check": row[0] == 1}


# --- Team: mount your routers below as you build them ---------------------
# from api.routes import router as api_router
# app.include_router(api_router, prefix="/api")
from api.auth import router as auth_router
app.include_router(auth_router, prefix="/auth", tags=["auth"])