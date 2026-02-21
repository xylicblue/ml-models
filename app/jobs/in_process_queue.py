"""In-process job queue using asyncio for local development.

Runs ML inference sequentially (one GPU job at a time) in a background task.
No external dependencies (Redis, Celery) needed.
"""

import asyncio
import traceback
from datetime import datetime
from typing import Callable, Dict, Optional

from app.jobs.dispatcher import JobDispatcher
from app.jobs.models import JobRecord, JobStatus


class InProcessQueue(JobDispatcher):
    """Local async job queue. Processes jobs one at a time via asyncio."""

    def __init__(self, worker_fn: Callable):
        """
        worker_fn: callable(job: JobRecord) -> JobRecord
            Synchronous function that does the work (runs model inference).
            Will be called in a thread executor to avoid blocking the event loop.
        """
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: Dict[str, JobRecord] = {}
        self._worker_fn = worker_fn
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def submit(self, job: JobRecord) -> str:
        self._jobs[job.id] = job
        await self._queue.put(job.id)
        return job.id

    async def get_status(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _worker_loop(self) -> None:
        """Process jobs one at a time from the queue."""
        while self._running:
            try:
                job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            job = self._jobs.get(job_id)
            if job is None:
                continue

            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()

            try:
                # Run the synchronous worker function in a thread executor
                # to avoid blocking the asyncio event loop during GPU inference
                loop = asyncio.get_event_loop()
                updated_job = await loop.run_in_executor(
                    None, self._worker_fn, job
                )
                self._jobs[job_id] = updated_job
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                job.completed_at = datetime.utcnow()
