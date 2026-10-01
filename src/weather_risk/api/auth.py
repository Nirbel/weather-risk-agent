"""Small email + password auth for the take-home (docs/DECISIONS.md D18).

- Sign-up is open only to addresses whose domain is exactly `moveo.co.il`, and always creates
  an `analyst`. The only admin is the seeded demo account (settings: DEMO_ADMIN_*).
- Passwords are stored as salted scrypt hashes (stdlib hashlib), never as text.
- Login returns a signed bearer token (HS256, 12 h). Every request re-reads the account, so the
  role comes from the database, not from the token, and a deleted account loses access at once.
- Deliberately not included: password reset, email verification, OAuth, refresh tokens.
"""

import base64
import hashlib
import hmac
import re
import secrets
import unicodedata
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Literal

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from weather_risk.db import Database
from weather_risk.settings import Settings

Role = Literal["analyst", "admin"]
SIGNUP_DOMAIN = "moveo.co.il"
MIN_PASSWORD, MAX_PASSWORD = 8, 128
# RFC 5322 "dot-atom" local part, ASCII only: no quotes, spaces, or leading/trailing/double dots.
_LOCAL_PART = re.compile(r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*")
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}  # ~16 MiB, tens of ms per hash
_bearer = HTTPBearer(auto_error=False)
INVALID_LOGIN = "Invalid email or password."


class User(BaseModel):
    email: str
    role: Role


# -- email ---------------------------------------------------------------------------

def normalize_email(raw: str) -> str:
    """Unicode NFKC (folds full-width look-alikes such as "＠"), trim, lower-case."""
    return unicodedata.normalize("NFKC", raw).strip().lower()


def signup_email_error(email: str) -> str | None:
    """None if a *normalized* email may sign up, else the reason. The domain is the part after the
    last "@" and must equal moveo.co.il exactly, so user@moveo.co.il.attacker.com is refused."""
    local, at, domain = email.rpartition("@")
    if not at or not local or not domain:
        return "Enter a valid email address."
    if domain != SIGNUP_DOMAIN:
        return f"Sign-up is limited to @{SIGNUP_DOMAIN} email addresses."
    if len(local) > 64 or not _LOCAL_PART.fullmatch(local):
        return "Enter a valid email address."
    return None


# -- passwords -----------------------------------------------------------------------

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def hash_password(password: str) -> str:
    """'scrypt$n$r$p$salt$hash' with a random 16-byte salt."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, n, r, p, salt, digest = stored.split("$")
        if algorithm != "scrypt":
            return False
        expected = base64.b64decode(digest, validate=True)
        candidate = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt, validate=True),
                                   n=int(n), r=int(r), p=int(p), dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return bool(expected) and hmac.compare_digest(candidate, expected)


@lru_cache
def _dummy_hash() -> str:
    return hash_password(secrets.token_urlsafe(16))


# -- tokens and accounts -------------------------------------------------------------

def issue_token(user: User, settings: Settings) -> str:
    now = datetime.now(UTC)
    claims = {"sub": user.email, "role": user.role, "iat": now, "exp": now + timedelta(hours=settings.token_ttl_hours)}
    return jwt.encode(claims, settings.jwt_secret, algorithm="HS256")


def check_password_length(password: str) -> None:
    if not MIN_PASSWORD <= len(password) <= MAX_PASSWORD:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            f"The password must be {MIN_PASSWORD}–{MAX_PASSWORD} characters.")


async def signup(db: Database, email: str, password: str, settings: Settings) -> tuple[User, str]:
    email = normalize_email(email)
    if error := signup_email_error(email):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, error)
    check_password_length(password)
    if not await db.create_account(email, hash_password(password), role="analyst"):  # never admin
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists — log in instead.")
    user = User(email=email, role="analyst")
    return user, issue_token(user, settings)


async def login(db: Database, email: str, password: str, settings: Settings) -> tuple[User, str]:
    account = await db.get_account(normalize_email(email))
    if account is None:
        verify_password(password, _dummy_hash())  # same work as a real check: no timing hint
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_LOGIN)
    if not verify_password(password, account["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_LOGIN)
    user = User(email=account["email"], role=account["role"])
    return user, issue_token(user, settings)


async def ensure_demo_admin(db: Database, settings: Settings) -> None:
    """Create the demo admin on first start. Its configured password is authoritative, so changing
    DEMO_ADMIN_PASSWORD and restarting rotates it. Sign-up can never create or claim this account."""
    if not (settings.demo_admin_email and settings.demo_admin_password):
        return
    email = normalize_email(settings.demo_admin_email)
    account = await db.get_account(email)
    if account is None:
        await db.create_account(email, hash_password(settings.demo_admin_password), role="admin")
    elif not verify_password(settings.demo_admin_password, account["password_hash"]):
        await db.set_password(email, hash_password(settings.demo_admin_password))


async def current_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> User:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token — log in first.")
    settings: Settings = request.app.state.settings
    try:
        claims = jwt.decode(creds.credentials, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {exc}") from exc
    account = await request.app.state.db.get_account(claims.get("sub", ""))
    if account is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "This account no longer exists.")
    return User(email=account["email"], role=account["role"])  # role from the database, not the token


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required.")
    return user


async def require_admin_or_trigger_secret(request: Request, x_alert_secret: str | None = Header(None),
                                          creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    """POST /alerts/run: an admin token, or the shared secret an external cron or webhook sends."""
    secret = request.app.state.settings.alert_trigger_secret
    if secret and x_alert_secret and hmac.compare_digest(x_alert_secret.encode(), secret.encode()):
        return "trigger-secret"
    return require_admin(await current_user(request, creds)).email
