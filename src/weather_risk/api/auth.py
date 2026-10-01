"""Light allow-list auth (docs/DECISIONS.md D18).

An allow-listed email gets a signed bearer token carrying its role (analyst | admin).
There is no password: this identifies users for a demo, it does not authenticate them.
"""

import hmac
from datetime import UTC, datetime, timedelta
from typing import Literal

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from weather_risk.settings import Settings

Role = Literal["analyst", "admin"]
_bearer = HTTPBearer(auto_error=False)


class User(BaseModel):
    email: str
    role: Role


def parse_allowlist(raw: str) -> dict[str, Role]:
    """'a@x.com:admin, b@y.com:analyst' → {email: role}. A missing role means analyst."""
    allowed: dict[str, Role] = {}
    for item in filter(None, (part.strip() for part in raw.split(","))):
        email, _, role = item.partition(":")
        role = role.strip().lower() or "analyst"
        if role not in ("analyst", "admin"):
            raise ValueError(f"unknown role {role!r} for {email}")
        allowed[email.strip().lower()] = role  # type: ignore[assignment]
    return allowed


def issue_token(user: User, settings: Settings) -> str:
    now = datetime.now(UTC)
    claims = {"sub": user.email, "role": user.role, "iat": now, "exp": now + timedelta(hours=settings.token_ttl_hours)}
    return jwt.encode(claims, settings.jwt_secret, algorithm="HS256")


def login(email: str, settings: Settings) -> tuple[User, str]:
    allowed = parse_allowlist(settings.auth_allowlist)
    if not allowed:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "No users are allowed yet — set AUTH_ALLOWLIST in .env (e.g. you@company.com:admin).")
    role = allowed.get(email.strip().lower())
    if role is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This email is not on the allow-list.")
    user = User(email=email.strip().lower(), role=role)
    return user, issue_token(user, settings)


def current_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> User:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token — log in first.")
    settings: Settings = request.app.state.settings
    try:
        claims = jwt.decode(creds.credentials, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {exc}") from exc
    # Re-check the allow-list so removing an email revokes access immediately.
    role = parse_allowlist(settings.auth_allowlist).get(claims["sub"])
    if role is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "This email is no longer on the allow-list.")
    return User(email=claims["sub"], role=role)


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required.")
    return user


def require_admin_or_trigger_secret(request: Request, x_alert_secret: str | None = Header(None),
                                    creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    """POST /alerts/run: an admin token, or the shared secret an external cron or webhook sends."""
    secret = request.app.state.settings.alert_trigger_secret
    if secret and x_alert_secret and hmac.compare_digest(x_alert_secret.encode(), secret.encode()):
        return "trigger-secret"
    return require_admin(current_user(request, creds)).email
