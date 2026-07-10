"""简历评估五维评分配置。"""

from collections import OrderedDict
from dataclasses import dataclass

DIMENSION_ORDER = (
    "technology_stack",
    "experience",
    "education",
    "potential",
    "resume_structure",
)

DEFAULT_WEIGHTS: "OrderedDict[str, int]" = OrderedDict(
    (
        ("technology_stack", 30),
        ("experience", 25),
        ("education", 15),
        ("potential", 15),
        ("resume_structure", 15),
    )
)


@dataclass(frozen=True)
class ScoringConfig:
    """简历评估评分权重配置。"""

    weights: OrderedDict[str, int]
    dimension_profile: str = "default_v1"
    warnings: tuple[str, ...] = ()

    @classmethod
    def default(cls) -> "ScoringConfig":
        """返回默认 V1 权重。"""
        return cls(weights=DEFAULT_WEIGHTS.copy())


def parse_weights(raw_value: str | None) -> OrderedDict[str, int]:
    """解析并校验权重配置，非法值 fail-closed。"""
    if raw_value is None or not raw_value.strip():
        return DEFAULT_WEIGHTS.copy()

    parsed: dict[str, int] = {}
    for item in raw_value.split(","):
        if ":" not in item:
            raise ValueError("AI_WORKER_RESUME_EVALUATION_WEIGHTS is invalid")
        name, raw_weight = item.split(":", 1)
        name = name.strip()
        raw_weight = raw_weight.strip()
        if name not in DIMENSION_ORDER:
            raise ValueError("AI_WORKER_RESUME_EVALUATION_WEIGHTS contains unknown dimension")
        try:
            weight = int(raw_weight)
        except ValueError as exc:
            raise ValueError("AI_WORKER_RESUME_EVALUATION_WEIGHTS contains non-integer weight") from exc
        if weight < 0:
            raise ValueError("AI_WORKER_RESUME_EVALUATION_WEIGHTS contains negative weight")
        parsed[name] = weight

    if tuple(parsed.keys()) != DIMENSION_ORDER:
        raise ValueError("AI_WORKER_RESUME_EVALUATION_WEIGHTS dimensions are incomplete or out of order")
    if sum(parsed.values()) != 100:
        raise ValueError("AI_WORKER_RESUME_EVALUATION_WEIGHTS sum must be 100")
    return OrderedDict((name, parsed[name]) for name in DIMENSION_ORDER)


def config_for_profile(profile: str | None, weights: OrderedDict[str, int] | None = None) -> ScoringConfig:
    """根据 profile 返回评分配置，未知 profile 降级为默认权重。"""
    if profile is None or not profile.strip() or profile == "default_v1":
        return ScoringConfig(weights=weights.copy() if weights is not None else DEFAULT_WEIGHTS.copy())
    return ScoringConfig(
        weights=DEFAULT_WEIGHTS.copy(),
        warnings=("unknown_dimension_profile",),
    )
