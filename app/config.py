"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Supabase
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_anon_key: str = ""

    # Image storage
    imagery_dir: str = "/data"
    model_weights_dir: str = "/app/model_weights"

    # Compute dispatch
    compute_dispatch_mode: str = "local"  # "local" or "modal"
    compute_port: int = 8001

    # Model registry
    max_loaded_models: int = 3

    # Job processing
    job_result_ttl_hours: int = 2
    max_image_dimension: int = 50000

    # Modal (only when dispatch_mode=modal)
    modal_token_id: Optional[str] = None
    modal_token_secret: Optional[str] = None

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
