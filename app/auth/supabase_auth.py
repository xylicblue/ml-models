"""Supabase JWT validation dependency for FastAPI."""

from fastapi import Header, HTTPException
from supabase import create_client
from app.config import settings


async def verify_jwt(authorization: str = Header(None)) -> dict:
    """Validate Supabase JWT from Authorization header.

    Returns the authenticated user object.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.replace("Bearer ", "")
    try:
        client = create_client(settings.supabase_url, settings.supabase_anon_key)
        user_response = client.auth.get_user(token)
        return user_response.user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
