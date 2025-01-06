#!/bin/bash
set -e

# Set the necessary configurations
echo "shared_preload_libraries = 'timescaledb,pgvector'" >> $PGDATA/postgresql.conf
echo "timescaledb.telemetry_level=off" >> $PGDATA/postgresql.conf

# Restart PostgreSQL to apply the configurations
pg_ctl restart

# Wait for PostgreSQL to be ready
until pg_isready -U postgres; do
  sleep 1
done

# Create the extensions
psql -U postgres -d postgres -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
psql -U postgres -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
