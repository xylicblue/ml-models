"""Temporary result storage for output images with auto-cleanup."""

import os
import time
import tempfile
from typing import Optional

from app.config import settings


class TempResultStore:
    """Manages temporary output files (visualization images) with TTL-based cleanup."""

    def __init__(self, base_dir: Optional[str] = None, ttl_hours: int = 2):
        if base_dir:
            self._base_dir = base_dir
        else:
            self._base_dir = os.path.join(tempfile.gettempdir(), "agripay_results")
        os.makedirs(self._base_dir, exist_ok=True)
        self._ttl_seconds = ttl_hours * 3600

    def get_job_dir(self, job_id: str) -> str:
        """Get or create directory for a job's output files."""
        job_dir = os.path.join(self._base_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        return job_dir

    def get_output_path(self, job_id: str, filename: str) -> str:
        """Get full path for a specific output file."""
        return os.path.join(self.get_job_dir(job_id), filename)

    def file_exists(self, job_id: str, filename: str) -> bool:
        return os.path.exists(self.get_output_path(job_id, filename))

    def cleanup_expired(self) -> int:
        """Remove job directories older than TTL. Returns count of removed dirs."""
        now = time.time()
        removed = 0
        if not os.path.exists(self._base_dir):
            return 0
        for entry in os.listdir(self._base_dir):
            job_dir = os.path.join(self._base_dir, entry)
            if not os.path.isdir(job_dir):
                continue
            mtime = os.path.getmtime(job_dir)
            if now - mtime > self._ttl_seconds:
                import shutil
                shutil.rmtree(job_dir, ignore_errors=True)
                removed += 1
        return removed


# Global instance
temp_store = TempResultStore(ttl_hours=settings.job_result_ttl_hours)
