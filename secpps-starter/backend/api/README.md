# api — owner: Jazim

REST routers (FastAPI `APIRouter`) mounted into `main.py`. Also: JWT auth,
WebSocket endpoint for the live dashboard.

## Suggested first files
- `auth.py` — register/login, JWT issue/verify
- `readings.py` — GET endpoints over `energy_readings` for the dashboard
- `ws.py` — WebSocket push of live usage (every 60s per the proposal)
