import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Runtime configuration for the public local-first app."""

    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1")
    DATA_MAX_AGE_HOURS = int(os.getenv("DATA_MAX_AGE_HOURS", "168"))
    DATA_CLEANUP_INTERVAL_MINUTES = int(os.getenv("DATA_CLEANUP_INTERVAL_MINUTES", "60"))


config = Config()
