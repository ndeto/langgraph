"""add client quota buckets and thread messages

Revision ID: 0003_client_quota_msgs
Revises: 0002_worker_jobs
Create Date: 2026-08-05 21:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_client_quota_msgs"
down_revision = "0002_worker_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "client_key_quotas",
        sa.Column("client_key", sa.String(), primary_key=True),
        sa.Column("questions_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("uploads_used", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_table(
        "thread_messages",
        sa.Column("message_id", sa.String(), primary_key=True),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("assets", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_thread_messages_thread_id", "thread_messages", ["thread_id"])


def downgrade() -> None:
    op.drop_index("ix_thread_messages_thread_id", table_name="thread_messages")
    op.drop_table("thread_messages")
    op.drop_table("client_key_quotas")
