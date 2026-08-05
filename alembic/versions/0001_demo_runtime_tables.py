"""create demo runtime tables

Revision ID: 0001_demo_runtime_tables
Revises: None
Create Date: 2026-08-04 14:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_demo_runtime_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "demo_sessions",
        sa.Column("user_id", sa.String(), primary_key=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("active_document", sa.String(), nullable=True),
        sa.Column("active_thread", sa.String(), nullable=True),
    )

    op.create_table(
        "documents",
        sa.Column("document_id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"])

    op.create_table(
        "ingestion_jobs",
        sa.Column("job_id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_ingestion_jobs_user_id", "ingestion_jobs", ["user_id"])
    op.create_index("ix_ingestion_jobs_document_id", "ingestion_jobs", ["document_id"])

    op.create_table(
        "ingestion_events",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.UniqueConstraint("job_id", "event_id", name="uq_ingestion_event_job_id"),
    )
    op.create_index("ix_ingestion_events_job_id", "ingestion_events", ["job_id"])

    op.create_table(
        "threads",
        sa.Column("thread_id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("document_id", sa.String(), nullable=True),
    )
    op.create_index("ix_threads_user_id", "threads", ["user_id"])

    op.create_table(
        "user_quotas",
        sa.Column("user_id", sa.String(), primary_key=True),
        sa.Column("questions_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("uploads_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("active_ingestions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("active_agent_runs", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_table(
        "request_key_quotas",
        sa.Column("request_key", sa.String(), primary_key=True),
        sa.Column("questions_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("uploads_used", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_table(
        "usage_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("operation", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("trace_id", sa.String(), nullable=True),
        sa.UniqueConstraint("user_id", "run_id", name="uq_usage_user_run"),
    )
    op.create_index("ix_usage_records_user_id", "usage_records", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_usage_records_user_id", table_name="usage_records")
    op.drop_table("usage_records")
    op.drop_table("request_key_quotas")
    op.drop_table("user_quotas")
    op.drop_index("ix_threads_user_id", table_name="threads")
    op.drop_table("threads")
    op.drop_index("ix_ingestion_events_job_id", table_name="ingestion_events")
    op.drop_table("ingestion_events")
    op.drop_index("ix_ingestion_jobs_document_id", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_user_id", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
    op.drop_index("ix_documents_user_id", table_name="documents")
    op.drop_table("documents")
    op.drop_table("demo_sessions")
