import logging
import sys
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional

from rag_platform.core.config import settings

logger = logging.getLogger(__name__)


class NoopObservation:
    def update(self, **kwargs: Any) -> None:
        return None

    def score(self, **kwargs: Any) -> None:
        return None


class LangfuseObservability:
    def __init__(self) -> None:
        self._client = None
        self._sdk_load_failed = False

    @property
    def enabled(self) -> bool:
        return bool(
            settings.langfuse_enabled
            and settings.langfuse_public_key
            and settings.langfuse_secret_key
            and settings.langfuse_sample_rate > 0
        )

    @contextmanager
    def trace(
        self,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Iterator[Any]:
        manager = self._start_observation(
            name=name,
            as_type="span",
            input_data=input_data,
            metadata=metadata,
        )
        observation = self._enter_manager(manager)
        if observation is None:
            yield NoopObservation()
            return

        self._update_current_trace(
            metadata=metadata,
            user_id=user_id,
            session_id=session_id,
            tags=tags,
            trace_name=name,
        )

        exc_info = (None, None, None)
        try:
            yield observation
        except Exception as exc:
            exc_info = sys.exc_info()
            self.update_observation(
                observation,
                level="ERROR",
                status_message=str(exc),
            )
            raise
        finally:
            self._exit_manager(manager, exc_info)
            self.flush()

    @contextmanager
    def observation(
        self,
        name: str,
        as_type: str = "span",
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Iterator[Any]:
        manager = self._start_observation(
            name=name,
            as_type=as_type,
            input_data=input_data,
            metadata=metadata,
            model=model,
        )
        observation = self._enter_manager(manager)
        if observation is None:
            yield NoopObservation()
            return

        exc_info = (None, None, None)
        try:
            yield observation
        except Exception as exc:
            exc_info = sys.exc_info()
            self.update_observation(
                observation,
                level="ERROR",
                status_message=str(exc),
            )
            raise
        finally:
            self._exit_manager(manager, exc_info)

    def update_observation(self, observation: Any, **payload: Any) -> None:
        if isinstance(observation, NoopObservation):
            return
        try:
            observation.update(**{key: value for key, value in payload.items() if value is not None})
        except Exception as exc:  # pragma: no cover - Langfuse SDK/network boundary
            logger.warning("Langfuse observation update failed: %s", exc)

    def score_observation(
        self,
        observation: Any,
        name: str,
        value: Any,
        comment: Optional[str] = None,
    ) -> None:
        if isinstance(observation, NoopObservation):
            return
        score = getattr(observation, "score", None)
        if not callable(score):
            return
        try:
            score(name=name, value=value, comment=comment)
        except Exception as exc:  # pragma: no cover - Langfuse SDK/network boundary
            logger.warning("Langfuse score upload failed: %s", exc)

    def flush(self) -> None:
        client = self._get_client()
        if client is None:
            return
        flush = getattr(client, "flush", None)
        if not callable(flush):
            return
        try:
            flush()
        except Exception as exc:  # pragma: no cover - Langfuse SDK/network boundary
            logger.warning("Langfuse flush failed: %s", exc)

    def _start_observation(
        self,
        name: str,
        as_type: str,
        input_data: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
        model: Optional[str] = None,
    ):
        client = self._get_client()
        if client is None:
            return None

        payload: Dict[str, Any] = {"as_type": as_type, "name": name}
        if input_data is not None:
            payload["input"] = input_data
        if model:
            payload["model"] = model

        try:
            manager = client.start_as_current_observation(**payload)
        except Exception as exc:  # pragma: no cover - Langfuse SDK/network boundary
            logger.warning("Langfuse observation start failed: %s", exc)
            return None
        return _ObservationManager(manager, metadata, self)

    def _update_current_trace(
        self,
        metadata: Optional[Dict[str, Any]],
        user_id: Optional[str],
        session_id: Optional[str],
        tags: Optional[list[str]],
        trace_name: Optional[str],
    ) -> None:
        client = self._get_client()
        if client is None:
            return None
        payload: Dict[str, Any] = {}
        if metadata:
            # Langfuse trace metadata 适合放低基数摘要字段，详细候选放 observation output。
            payload["metadata"] = self._metadata_strings(metadata)
        if user_id:
            payload["user_id"] = user_id
        if session_id:
            payload["session_id"] = session_id
        if tags:
            payload["tags"] = tags
        if trace_name:
            payload["name"] = trace_name
        if not payload:
            return None
        try:
            client.update_current_trace(**payload)
        except Exception as exc:  # pragma: no cover - Langfuse SDK boundary
            logger.warning("Langfuse trace update failed: %s", exc)
        return None

    def _metadata_strings(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        normalized = {}
        for key, value in metadata.items():
            if value is None:
                continue
            normalized[str(key)] = str(value)[:200]
        return normalized

    def _get_client(self):
        if not self.enabled or self._sdk_load_failed:
            return None
        if self._client is not None:
            return self._client

        try:
            from langfuse import Langfuse
        except ImportError as exc:  # pragma: no cover - environment guard
            self._sdk_load_failed = True
            logger.warning("Langfuse SDK is not installed: %s", exc)
            return None

        kwargs: Dict[str, Any] = {
            "public_key": settings.langfuse_public_key,
            "secret_key": settings.langfuse_secret_key,
            "sample_rate": settings.langfuse_sample_rate,
        }
        host = settings.langfuse_host or settings.langfuse_base_url
        if host:
            kwargs["host"] = host
        if settings.langfuse_environment:
            kwargs["environment"] = settings.langfuse_environment

        try:
            self._client = Langfuse(**kwargs)
        except Exception as exc:  # pragma: no cover - Langfuse SDK boundary
            self._sdk_load_failed = True
            logger.warning("Langfuse client initialization failed: %s", exc)
            return None
        return self._client

    def _enter_manager(self, manager: Any):
        if manager is None:
            return None
        try:
            return manager.__enter__()
        except Exception as exc:  # pragma: no cover - Langfuse SDK boundary
            logger.warning("Langfuse context enter failed: %s", exc)
            return None

    def _exit_manager(self, manager: Any, exc_info: tuple[Any, Any, Any]) -> None:
        if manager is None:
            return
        try:
            manager.__exit__(*exc_info)
        except Exception as exc:  # pragma: no cover - Langfuse SDK boundary
            logger.warning("Langfuse context exit failed: %s", exc)


class _ObservationManager:
    def __init__(
        self,
        manager: Any,
        metadata: Optional[Dict[str, Any]],
        observability: LangfuseObservability,
    ) -> None:
        self.manager = manager
        self.metadata = metadata
        self.observability = observability
        self.observation = None

    def __enter__(self):
        self.observation = self.manager.__enter__()
        if self.metadata:
            self.observability.update_observation(self.observation, metadata=self.metadata)
        return self.observation

    def __exit__(self, exc_type, exc, tb):
        return self.manager.__exit__(exc_type, exc, tb)


@lru_cache
def get_langfuse_observability() -> LangfuseObservability:
    return LangfuseObservability()
