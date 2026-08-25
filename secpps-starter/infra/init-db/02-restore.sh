#!/bin/bash
# Runs automatically the FIRST time the db container starts on an empty volume.
# Restores Alka's populated database from smart_energy.dump, which must sit
# next to this script (same infra/init-db/ folder).
set -e

if [ -f /docker-entrypoint-initdb.d/smart_energy.dump ]; then
  echo "Restoring smart_energy.dump into $POSTGRES_DB ..."
  pg_restore --no-owner --role="$POSTGRES_USER" -U "$POSTGRES_USER" -d "$POSTGRES_DB" /docker-entrypoint-initdb.d/smart_energy.dump
  echo "Restore complete."
else
  echo "No smart_energy.dump found in infra/init-db/ — skipping restore. Database will be empty."
fi