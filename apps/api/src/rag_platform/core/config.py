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
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_endpoint_path: str = ""
    embedding_api_key: str = ""
    embedding_dim: int = 0
    google_api_key: str = ""
    llm_provider: str = ""
    llm_openai_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_max_output_tokens: int = 0
    hyde_openai_base_url: str = ""
    hyde_llm_model: str = ""
    hyde_max_output_tokens: int = 0
    openai_api_key: str = ""
    ragas_openai_api_key: str = ""
    ragas_openai_base_url: str = ""
    ragas_llm_model: str = ""
    ragas_embedding_model: str = ""
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""
    langfuse_base_url: str = ""
    langfuse_environment: str = ""
    langfuse_sample_rate: float = Field(default=1.0, ge=0, le=1)
    web_origin: str = "http://localhost:5173"
    upload_dir: Path = Field(default=Path("uploads"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
