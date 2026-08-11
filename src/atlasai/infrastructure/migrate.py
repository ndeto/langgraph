"""Guarded production database migration entry point."""

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select

from atlasai.config.sys_config import get_env
from atlasai.infrastructure.postgres_repositories import ingestion_jobs_table


def active_ingestion_count() -> int:
    """Return queued or processing ingestion jobs when the table exists."""

    engine = create_engine(get_env("DB_CONN"), future=True)
    try:
        with engine.connect() as connection:
            if not inspect(connection).has_table(ingestion_jobs_table.name):
                return 0
            statement = select(ingestion_jobs_table.c.job_id).where(
                ingestion_jobs_table.c.state.in_(("queued", "processing"))
            )
            return len(connection.execute(statement).all())
    finally:
        engine.dispose()


def main() -> None:
    """Refuse migration during ingestion, then upgrade to the latest revision."""

    active_jobs = active_ingestion_count()
    if active_jobs:
        raise SystemExit(
            f"Migration blocked: {active_jobs} ingestion job(s) are queued or processing."
        )

    command.upgrade(Config("alembic.ini"), "head")


if __name__ == "__main__":
    main()
