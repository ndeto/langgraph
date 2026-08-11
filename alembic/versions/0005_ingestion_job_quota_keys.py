"""store quota bucket keys on ingestion jobs

Revision ID: 0005_ingestion_job_quota_keys
Revises: 0004_postgres_assets
Create Date: 2026-08-06 09:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0005_ingestion_job_quota_keys"
down_revision = "0004_postgres_assets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ingestion_jobs", sa.Column("client_key", sa.String(), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("ip_hash", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("ingestion_jobs", "ip_hash")
    op.drop_column("ingestion_jobs", "client_key")
