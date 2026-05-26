import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

import httpx

from rag_platform.core.config import settings


class WebhookConfigurationError(Exception):
    pass


class WebhookAuthError(Exception):
    pass


class WebhookPayloadError(Exception):
    pass


class GitHubDispatchError(Exception):
    pass


def verify_langfuse_signature(
    raw_body: bytes,
    signature_header: str,
    secret: str,
    tolerance_seconds: int = 300,
    now: Optional[float] = None,
) -> bool:
    if not raw_body or not signature_header or not secret:
        return False
    try:
        timestamp_pair, signature_pair = signature_header.split(",", 1)
        timestamp = timestamp_pair.split("=", 1)[1]
        received_signature = signature_pair.split("=", 1)[1]
    except (IndexError, ValueError):
        return False

    try:
        timestamp_seconds = int(timestamp)
    except ValueError:
        return False
    current_time = int(now if now is not None else time.time())
    if tolerance_seconds > 0 and abs(current_time - timestamp_seconds) > tolerance_seconds:
        return False

    try:
        raw_text = raw_body.decode("utf-8")
    except UnicodeDecodeError:
        return False
    message = f"{timestamp}.{raw_text}".encode("utf-8")
    expected_signature = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    try:
        return hmac.compare_digest(
            bytes.fromhex(received_signature),
            bytes.fromhex(expected_signature),
        )
    except ValueError:
        return False


class PromptCICDService:
    async def handle_prompt_webhook(
        self,
        raw_body: bytes,
        signature_header: str,
    ) -> Dict[str, Any]:
        self._verify_webhook(raw_body, signature_header)
        payload = self._load_payload(raw_body)
        dispatch = await self._dispatch_to_github(payload)
        prompt = payload.get("prompt", {}) if isinstance(payload.get("prompt"), dict) else {}
        return {
            "status": "accepted",
            "prompt": {
                "name": prompt.get("name"),
                "version": prompt.get("version"),
                "labels": prompt.get("labels", []),
            },
            "dispatch": dispatch,
        }

    def _verify_webhook(self, raw_body: bytes, signature_header: str) -> None:
        if not settings.langfuse_prompt_webhook_secret:
            raise WebhookConfigurationError("LANGFUSE_PROMPT_WEBHOOK_SECRET is not configured")
        if not verify_langfuse_signature(
            raw_body=raw_body,
            signature_header=signature_header,
            secret=settings.langfuse_prompt_webhook_secret,
            tolerance_seconds=settings.langfuse_prompt_webhook_tolerance_seconds,
        ):
            raise WebhookAuthError("invalid Langfuse webhook signature")

    def _load_payload(self, raw_body: bytes) -> Dict[str, Any]:
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WebhookPayloadError("webhook body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise WebhookPayloadError("webhook body must be a JSON object")
        if payload.get("type") != "prompt-version" or not isinstance(payload.get("prompt"), dict):
            raise WebhookPayloadError("webhook payload must be a Langfuse prompt-version event")
        return payload

    async def _dispatch_to_github(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not settings.github_repository_dispatch_url or not settings.github_repository_dispatch_token:
            return {"status": "skipped", "reason": "GitHub repository_dispatch is not configured"}

        body = {
            "event_type": settings.github_repository_dispatch_event_type,
            "client_payload": payload,
        }
        headers = {
            "Authorization": f"Bearer {settings.github_repository_dispatch_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                settings.github_repository_dispatch_url,
                headers=headers,
                json=body,
            )
        if response.status_code not in {200, 201, 202, 204}:
            raise GitHubDispatchError(
                f"GitHub repository_dispatch failed with status {response.status_code}"
            )
        return {
            "status": "sent",
            "event_type": settings.github_repository_dispatch_event_type,
            "status_code": response.status_code,
        }
