"""Job record data model for async processing."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecord(BaseModel):
    """Tracks the lifecycle of an async processing job."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    status: JobStatus = JobStatus.PENDING
    progress_current: int = 0
    progress_total: int = 0
    progress_message: str = ""
    input_params: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    output_files: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    user_id: Optional[str] = None
