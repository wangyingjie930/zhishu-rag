import uuid
from dataclasses import dataclass


DEMO_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
DEMO_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000101")


@dataclass(frozen=True)
class RequestContext:
    tenant_id: uuid.UUID = DEMO_TENANT_ID
    user_id: uuid.UUID = DEMO_USER_ID
    role: str = "admin"


async def get_request_context() -> RequestContext:
    # Production: replace with OIDC/JWT, tenant resolution, RBAC/ABAC and row-level policies.
    return RequestContext()

