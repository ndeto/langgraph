import logging
import os


def configure_logging() -> None:
    """Enable env-driven application logging for local runs."""

    level_name = os.getenv("ATLASAI_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    logging.getLogger("atlasai").setLevel(level)
