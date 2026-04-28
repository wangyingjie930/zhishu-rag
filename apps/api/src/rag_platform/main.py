from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_platform.api.v1.router import api_router
from rag_platform.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Enterprise knowledge base RAG platform",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.web_origin, "http://localhost:5173"],
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):51\d{2}$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
