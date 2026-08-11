"""store extracted document images in Postgres

Revision ID: 0004_postgres_assets
Revises: 0003_client_quota_msgs
Create Date: 2026-08-05 23:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0004_postgres_assets"
down_revision = "0003_client_quota_msgs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "assets",
        sa.Column("asset_id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("mime_type", sa.String(), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "document_id",
            "checksum",
            name="uq_assets_document_checksum",
        ),
    )
    op.create_index("ix_assets_user_id", "assets", ["user_id"])
    op.create_index("ix_assets_document_id", "assets", ["document_id"])
    op.create_index("ix_assets_expires_at", "assets", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_assets_expires_at", table_name="assets")
    op.drop_index("ix_assets_document_id", table_name="assets")
    op.drop_index("ix_assets_user_id", table_name="assets")
    op.drop_table("assets")
