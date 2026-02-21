"""Job dispatcher interface and in-process implementation."""

from abc import ABC, abstractmethod
from typing import Optional

from app.jobs.models import JobRecord


class JobDispatcher(ABC):
    """Abstract interface for job dispatching (local or cloud)."""

    @abstractmethod
    async def submit(self, job: JobRecord) -> str:
        """Submit a job for processing. Returns job_id."""
        ...

    @abstractmethod
    async def get_status(self, job_id: str) -> Optional[JobRecord]:
        """Get current status of a job."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the dispatcher (e.g., start worker loop)."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the dispatcher gracefully."""
        ...
