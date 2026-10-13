from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select

from backend.app.db import User
from backend.app.deps import Services, current_user, get_services
from backend.app.errors import ApiError
from backend.app.schemas import Credentials, TokenOut, UserOut
from backend.app.security import create_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_for(user: User, services: Services) -> TokenOut:
    token, expires_in = create_token(user.id, services.settings.secret_key, services.settings.token_minutes)
    return TokenOut(access_token=token, expires_in=expires_in, user=UserOut.model_validate(user))


@router.post("/register", response_model=TokenOut, status_code=201)
def register(body: Credentials, services: Services = Depends(get_services)) -> TokenOut:
    email = body.email.lower()
    with services.database.session() as session:
        exists = session.scalar(select(User).where(func.lower(User.email) == email))
        if exists is not None:
            raise ApiError(
                409, "email_taken", "An account with this email already exists. Log in instead.", "email"
            )
        user = User(email=email, password_hash=hash_password(body.password))
        session.add(user)
        session.flush()
        return _token_for(user, services)


@router.post("/login", response_model=TokenOut)
def login(body: Credentials, services: Services = Depends(get_services)) -> TokenOut:
    with services.database.session() as session:
        user = session.scalar(select(User).where(func.lower(User.email) == body.email.lower()))
    if user is None or not verify_password(body.password, user.password_hash):
        raise ApiError(401, "wrong_credentials", "Wrong email or password.")
    return _token_for(user, services)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> User:
    return user
