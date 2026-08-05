"""add worker job state columns

Revision ID: 0002_worker_jobs
Revises: 0001_demo_runtime_tables
Create Date: 2026-08-05 17:05:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_worker_jobs"
down_revision = "0001_demo_runtime_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column("source_path", sa.String(), nullable=True),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("failure_text", sa.String(), nullable=True),
    )

    op.create_table(
        "cleanup_jobs",
        sa.Column("job_id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("failure_text", sa.String(), nullable=True),
    )
    op.create_index("ix_cleanup_jobs_user_id", "cleanup_jobs", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_cleanup_jobs_user_id", table_name="cleanup_jobs")
    op.drop_table("cleanup_jobs")

    op.drop_column("ingestion_jobs", "failure_text")
    op.drop_column("ingestion_jobs", "updated_at")
    op.drop_column("ingestion_jobs", "heartbeat_at")
    op.drop_column("ingestion_jobs", "attempts")
    op.drop_column("ingestion_jobs", "source_path")
