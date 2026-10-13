"""Request and response bodies. Validation messages are shown to users as-is."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)

MAX_EPSILON = 32 / 255


def _check_epsilon(value: float) -> float:
    if not 0 <= value <= MAX_EPSILON:
        raise ValueError(f"epsilon must be between 0 and 32/255 ({MAX_EPSILON:.4f}); got {value:g}")
    return value


Epsilon = Annotated[float, AfterValidator(_check_epsilon)]


# ----------------------------------------------------------------------------- auth


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: UserOut


# ----------------------------------------------------------------------------- attacks


class Prediction(BaseModel):
    label: str
    index: int
    confidence: float


class AttackRequest(BaseModel):
    model_id: str
    dataset: str = "cifar10"
    epsilon: Epsilon = 8 / 255
    alpha: float | None = Field(default=None, description="PGD step size. Defaults to 2.5 * epsilon / steps.")
    steps: int = Field(default=10, ge=1, le=100)
    random_start: bool = True
    sample_indices: list[int] | None = Field(default=None, min_length=1)
    image: str | None = Field(default=None, description="Base64 PNG or JPEG, used instead of a test image.")
    label: str | None = Field(default=None, description="True class of an uploaded image, if known.")

    @field_validator("alpha")
    @classmethod
    def _alpha(cls, value: float | None) -> float | None:
        if value is not None and not 0 < value <= MAX_EPSILON:
            raise ValueError(f"alpha must be above 0 and at most 32/255 ({MAX_EPSILON:.4f})")
        return value

    @model_validator(mode="after")
    def _one_source(self) -> AttackRequest:
        if (self.sample_indices is None) == (self.image is None):
            raise ValueError("Give either sample_indices (test images) or image (an upload), not both")
        return self


class AttackItem(BaseModel):
    sample_index: int | None
    true_label: str | None
    clean_image: str
    adversarial_image: str
    perturbation_image: str
    heatmap_image: str
    clean_prediction: Prediction
    adversarial_prediction: Prediction
    clean_top: list[Prediction]
    adversarial_top: list[Prediction]
    fooled: bool
    linf: float
    l2: float


class AttackResponse(BaseModel):
    attack: Literal["fgsm", "pgd"]
    model_id: str
    epsilon: float
    alpha: float
    steps: int
    random_start: bool
    seconds: float
    success_per_step: list[float]
    success_rate: float
    items: list[AttackItem]


# ----------------------------------------------------------------------------- jobs


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    status: str
    params: dict
    progress: float
    message: str
    history: list
    result: dict | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class TrainingRequest(BaseModel):
    name: str | None = Field(default=None, max_length=60)
    model: Literal["small_cnn", "resnet18"] = "small_cnn"
    mode: Literal["clean", "fgsm", "pgd", "gaussian"] = "pgd"
    epsilon: Epsilon = 2 / 255
    pgd_steps: int = Field(default=3, ge=1, le=20)
    adv_ratio: float = Field(default=1.0, ge=0, le=1)
    sigma: float = Field(default=0.25, gt=0, le=1)
    epochs: int = Field(default=2, ge=1)
    batch_size: int = Field(default=128, ge=8, le=512)
    lr: float = Field(default=1e-3, gt=0, le=1)
    optimizer: Literal["adamw", "sgd"] = "adamw"
    max_train_batches: int | None = Field(default=40, ge=1)
    val_per_class: int = Field(default=20, ge=1, le=100)


class EvaluationRequest(BaseModel):
    model_ids: list[str] = Field(min_length=1, max_length=6)
    dataset: str = "cifar10"
    epsilons: list[Epsilon] = Field(default=[2 / 255, 4 / 255, 8 / 255], min_length=1, max_length=8)
    pgd_steps: int = Field(default=10, ge=1, le=50)
    threshold: float = Field(default=0.8, ge=0, le=1)
    images_per_class: int = Field(default=20, ge=1)

    @field_validator("epsilons")
    @classmethod
    def _sorted(cls, values: list[float]) -> list[float]:
        return sorted(set(values))

    @field_validator("model_ids")
    @classmethod
    def _unique(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class CertifyRequest(BaseModel):
    model_id: str
    dataset: str = "cifar10"
    sigma: float = Field(default=0.25, gt=0, le=1)
    n0: int = Field(default=25, ge=10, le=200)
    n: int = Field(default=100, ge=20, le=2000)
    alpha: float = Field(default=0.001, gt=0, lt=0.5)
    images_per_class: int = Field(default=5, ge=1, le=50)
