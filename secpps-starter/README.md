# SECPPS starter skeleton

This gives you a working `docker compose up` on day one: Postgres+TimescaleDB,
a FastAPI backend stub, and a placeholder frontend service — plus owner-labelled
folders matching your Week 1 role split.

## You already have two GitHub repos (frontend, backend) — how to merge this in

**Simplest path (recommended for a 12-week uni project):** stop maintaining
two repos and merge into one. Two repos is the wrong tradeoff for a 5-person
team on a tight timeline — it doubles PR overhead and makes `docker compose up`
harder for no real benefit at this scale.

1. Create one new repo, e.g. `secpps`.
2. Copy the contents of this starter into it as the base.
3. Copy your existing frontend repo's files into `frontend/` (overwrite the
   placeholder Dockerfile only if yours differs).
4. Copy your existing backend repo's files into `backend/` (merge with the
   stub `main.py` — keep the `/health` and `/health/db` routes, they're
   useful for debugging).
5. Archive (don't delete) the two old repos so history isn't lost.

**If you'd rather keep them separate:** clone both repos as sibling folders
next to this one (`../frontend`, `../backend`), delete this starter's
`frontend/` and `backend/` folders, and point `docker-compose.yml`'s `build.context`
at `../frontend` and `../backend` instead. More friction, but works.

## Running it

```bash
cp .env.example .env
docker compose up --build
```

Then check:
- Backend health: http://localhost:8000/health
- Backend + DB health: http://localhost:8000/health/db
- Frontend: http://localhost:5173

If `frontend/` doesn't have a real app yet (no `package.json`), the frontend
container will fail to build — that's expected until Jazim's React app lands.
Comment out the `frontend` service in `docker-compose.yml` until then so the
rest of the stack still comes up.

## Folder ownership (matches your Week 1 plan)

| Folder | Owner |
|---|---|
| `backend/data_pipeline/` | Alka |
| `backend/ml/features/` | Mishael |
| `backend/ml/anomaly/` | Mishael |
| `backend/ml/forecasting/` | Binita |
| `backend/api/` | Jazim |
| `backend/app_services/` | Abrein |
| `frontend/` | Jazim |
| `infra/init-db/01-init.sql` | Alka (schema) |
| `docs/interfaces.md` | Everyone — fill in together first |

## Next steps
1. Run the 30-min interface sync, fill in `docs/interfaces.md`.
2. Alka: get a slice of the SGSC dataset into `energy_readings` via
   `data_pipeline/ingest_csv.py` (create this file).
3. Everyone else: start building against the stub structure — you don't need
   to wait for the pipeline to be "done," just for *some* rows in the table.
