from fastapi import APIRouter, Header, HTTPException, Request, status

from rag_platform.services.prompt_cicd import (
    GitHubDispatchError,
    PromptCICDService,
    WebhookAuthError,
    WebhookConfigurationError,
    WebhookPayloadError,
)


router = APIRouter()


@router.post("/prompt-webhook", status_code=status.HTTP_202_ACCEPTED)
async def langfuse_prompt_webhook(
    request: Request,
    x_langfuse_signature: str = Header(default=""),
) -> dict:
    service = PromptCICDService()
    try:
        return await service.handle_prompt_webhook(
            raw_body=await request.body(),
            signature_header=x_langfuse_signature,
        )
    except WebhookConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except WebhookAuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except WebhookPayloadError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GitHubDispatchError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
