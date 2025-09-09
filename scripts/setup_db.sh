#!/usr/bin/env bash
set -euo pipefail

# ---- Configurable (or override via env) ----
DB_SUPERUSER=${DB_SUPERUSER:-postgres}       # OS auth user to run psql as (usually 'postgres')
DB_NAME=${DB_NAME:-riskintel}
DB_USER=${DB_USER:-riskintel_user}
DB_PASS=${DB_PASS:-riskintel2025}

# Use ON_ERROR_STOP so psql exits on the first error
PSQL_SUPER="psql -v ON_ERROR_STOP=1 -U ${DB_SUPERUSER} -d postgres"
PSQL_DBUSER="psql -v ON_ERROR_STOP=1 -U ${DB_USER} -d ${DB_NAME}"

echo "==> Creating role and database if they do not exist..."
$PSQL_SUPER <<SQL
DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN
      CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}';
   END IF;
END
\$\$;

DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}') THEN
      CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};
   END IF;
END
\$\$;
SQL

echo "==> Applying schema (scripts/db_init.sql) as ${DB_USER}..."
PGPASSWORD=${DB_PASS} psql -h 127.0.0.1 -v ON_ERROR_STOP=1 -U ${DB_USER} -d ${DB_NAME} -f scripts/db_init.sql


echo "==> Ensuring privileges inside ${DB_NAME}..."
psql -v ON_ERROR_STOP=1 -U ${DB_SUPERUSER} -d ${DB_NAME} <<SQL
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
GRANT USAGE, CREATE ON SCHEMA public TO ${DB_USER};
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${DB_USER};
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${DB_USER};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ${DB_USER};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ${DB_USER};
SQL

echo "==> Done."

