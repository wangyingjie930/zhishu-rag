from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    app_name: str = "Enterprise Knowledge RAG"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    sync_database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    object_storage_endpoint: str = "http://localhost:9000"
    object_storage_bucket: str = "rag-documents"
    object_storage_access_key: str = "minio"
    object_storage_secret_key: str = "minio123"
    embedding_model: str = "google:gemini-embedding-001"
    embedding_dim: int = 1536
    google_api_key: str = ""
    llm_provider: str = "mock"
    openai_api_key: str = ""
    web_origin: str = "http://localhost:5173"
    upload_dir: Path = Field(default=Path("uploads"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
