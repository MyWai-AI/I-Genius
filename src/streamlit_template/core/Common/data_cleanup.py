"""
Background data cleanup service.

Periodically deletes runtime data (downloaded videos, extracted frames,
pipeline results) from data/BAG, data/Generic, and data/SVO directories.
Never touches data/Common (robot models, AI models baked into the image).

Runs as a standalone process managed by supervisor alongside Streamlit.
"""

import logging
import os
import shutil
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[data_cleanup] %(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
MAX_AGE_HOURS = int(os.getenv("DATA_MAX_AGE_HOURS", "168"))  # 1 week
CLEANUP_INTERVAL_MINUTES = int(os.getenv("DATA_CLEANUP_INTERVAL_MINUTES", "60"))

# Runtime directories to clean (relative to DATA_ROOT)
RUNTIME_DIRS = ["BAG", "Generic", "SVO"]

# Directories that must NEVER be touched
PROTECTED_DIRS = {"Common"}


def _age_hours(path: Path) -> float:
    """Return file age in hours based on modification time."""
    try:
        return (time.time() - path.stat().st_mtime) / 3600
    except OSError:
        return 0


def cleanup_runtime_data() -> None:
    """
    Delete files and empty subdirectories older than MAX_AGE_HOURS
    inside each RUNTIME_DIR.  Recreate the top-level subdirectories
    so the application doesn't error on missing folders.
    """
    cutoff_hours = MAX_AGE_HOURS
    total_deleted = 0
    total_bytes = 0

    for dir_name in RUNTIME_DIRS:
        runtime_path = DATA_ROOT / dir_name
        if not runtime_path.exists():
            continue

        # Walk each session / subfolder inside the runtime dir
        for sub in sorted(runtime_path.iterdir()):
            if not sub.is_dir():
                # Top-level files inside runtime dir
                if _age_hours(sub) > cutoff_hours:
                    size = sub.stat().st_size
                    sub.unlink(missing_ok=True)
                    total_deleted += 1
                    total_bytes += size
                continue

            # Check if the entire subfolder is old enough to delete
            # Use the most recent modification time of any file inside
            newest_mtime = 0
            file_count = 0
            dir_size = 0

            for f in sub.rglob("*"):
                if f.is_file():
                    try:
                        st = f.stat()
                        newest_mtime = max(newest_mtime, st.st_mtime)
                        file_count += 1
                        dir_size += st.st_size
                    except OSError:
                        pass

            if file_count == 0:
                # Empty directory — remove it
                shutil.rmtree(sub, ignore_errors=True)
                continue

            age_h = (time.time() - newest_mtime) / 3600 if newest_mtime > 0 else 0

            if age_h > cutoff_hours:
                logger.info(
                    "Removing %s (%d files, %.1f MB, %.1f hours old)",
                    sub,
                    file_count,
                    dir_size / (1024 * 1024),
                    age_h,
                )
                shutil.rmtree(sub, ignore_errors=True)
                total_deleted += file_count
                total_bytes += dir_size

        # Recreate standard subdirectories so the app doesn't break
        runtime_path.mkdir(parents=True, exist_ok=True)

    if total_deleted > 0:
        logger.info(
            "Cleanup complete: removed %d files (%.1f MB)",
            total_deleted,
            total_bytes / (1024 * 1024),
        )
    else:
        logger.info("Cleanup complete: nothing to remove.")


def main():
    """Main loop — run cleanup periodically."""
    logger.info(
        "Data cleanup service started (max_age=%dh, interval=%dm, dirs=%s)",
        MAX_AGE_HOURS,
        CLEANUP_INTERVAL_MINUTES,
        RUNTIME_DIRS,
    )

    while True:
        try:
            cleanup_runtime_data()
        except Exception:
            logger.exception("Cleanup cycle failed")

        time.sleep(CLEANUP_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()
