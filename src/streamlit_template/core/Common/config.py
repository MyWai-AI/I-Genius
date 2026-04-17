import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for MyWai tool."""
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1")
    MYWAI_ENDPOINT = None
    
    if not DEBUG_MODE:
        MYWAI_ENDPOINT = os.getenv("END_POINT", None)
        if MYWAI_ENDPOINT is not None:
            MYWAI_ENDPOINT = MYWAI_ENDPOINT.lower()

    # Data cleanup settings
    DATA_MAX_AGE_HOURS = int(os.getenv("DATA_MAX_AGE_HOURS", "168"))  # 1 week
    DATA_CLEANUP_INTERVAL_MINUTES = int(os.getenv("DATA_CLEANUP_INTERVAL_MINUTES", "60"))



config = Config()
