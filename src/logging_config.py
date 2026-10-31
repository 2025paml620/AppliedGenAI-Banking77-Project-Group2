"""Small shared logging setup for command-line and demo entry points."""

import logging
import os


def configure_logging() -> None:
    """Set a consistent console format without recording customer text or secrets."""
    level_name = os.getenv("BANKING77_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
