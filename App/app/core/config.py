# app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional # Import Optional for optional fields
from pydantic import Field # Import Field for more control over defaults if needed

class Settings(BaseSettings):
    # Make optional fields truly optional or provide defaults
    ENV: Optional[str] = "development"

    # Database connection details - make them Optional if not strictly required
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[int] = None
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_NAME: Optional[str] = None

    # Google OAuth settings - make them Optional
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REDIRECT_URI: Optional[str] = None

    # Generative AI settings - make them Optional
    GENAI_API_KEY: Optional[str] = None
    GENAI_MODEL: Optional[str] = None

    # JWT Settings:
    # Remove the direct default values here if they are ALWAYS expected from .env
    # Pydantic-settings will now look for them in the .env file first.
    # If you absolutely need a fallback Python default *and* env var override,
    # you can use Field(default="value") or Field(default_factory=...)
    SECRET_KEY: str # No default here, must be in .env or env vars
    ALGORITHM: str = "HS256" # This default is fine if it's always "HS256" unless overridden
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30 # This default is fine if it's always 30 unless overridden
    DATABASE_URL: str # No default here, must be in .env or env vars


    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore" # Should allow extra fields in .env not defined in model
    )

@lru_cache()
def get_settings():
    # Pydantic will now correctly load values from .env for SECRET_KEY and DATABASE_URL
    # if they are present there, and for others, it will use class defaults or env vars.
    return Settings()