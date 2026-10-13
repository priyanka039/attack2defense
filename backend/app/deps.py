"""Shared services and the login dependencies used by the routers."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.db import Database, User
from backend.app.errors import ApiError
from backend.app.security import read_token
from backend.app.services.datasets import DatasetService
from backend.app.services.jobs import JobRunner
from backend.app.services.model_store import ModelStore
from backend.app.settings import Settings

_bearer = HTTPBearer(auto_error=False)


@dataclass
class Services:
    settings: Settings
    database: Database
    datasets: DatasetService
    models: ModelStore
    jobs: JobRunner


def get_services(request: Request) -> Services:
    return request.app.state.services


def optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    services: Services = Depends(get_services),
) -> User | None:
    if credentials is None:
        return None
    user_id = read_token(credentials.credentials, services.settings.secret_key)
    if user_id is None:
        raise ApiError(401, "invalid_token", "Your session has expired. Log in again.")
    with services.database.session() as session:
        user = session.get(User, user_id)
    if user is None:
        raise ApiError(401, "invalid_token", "This account no longer exists. Log in again.")
    return user


def current_user(user: User | None = Depends(optional_user)) -> User:
    if user is None:
        raise ApiError(401, "not_authenticated", "Log in to start attacks, training, or evaluations.")
    return user
