"""本地指纹、相似度与去重层级。

只依赖输入中的数值指纹与家族标识，不下载任何视觉模型。
所有指纹都可由输入稳定复算，便于把分析结果连回原稿与指纹记录。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from design_rights_chain.custody.case_store import stable_json


class StudyInputError(ValueError):
    """分析输入缺少必要字段或字段口径不合法。"""


def parse_instant(value: Any) -> str:
    """校验 ISO 8601 带时区时刻，返回标准化字符串。

    业务口径要求时刻均带时区；无时区信息会被拒绝，避免把本地时间
    与 UTC 混算后错误判断候选在查询时刻是否已入库。
    """
    if not isinstance(value, str) or not value:
        raise StudyInputError("时间字段必须是非空 ISO 8601 字符串")
    text = value.strip()
    if not (text.endswith("Z") or text[-6] in ("+", "-") or text[-5] in ("+", "-")):
        raise StudyInputError(f"时间字段缺少时区信息: {value}")
    try:
        normalized = text.replace("Z", "+00:00")
        from datetime import datetime

        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise StudyInputError(f"无法解析时间字段: {value}") from exc
    return normalized


def coerce_features(value: Any, owner: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise StudyInputError(f"{owner} 的 features 必须是非空数值数组")
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise StudyInputError(f"{owner} 的 features 只能包含数值")
        number = float(item)
        if not math.isfinite(number):
            raise StudyInputError(f"{owner} 的 features 含非有限数值")
        out.append(number)
    return tuple(out)


def fingerprint_vector(features: Sequence[float]) -> str:
    """对数值指纹做稳定哈希，作为本地指纹记录编号。"""
    return "fp-" + hashlib.sha256(stable_json(list(features)).encode("utf-8")).hexdigest()[:16]


def euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise StudyInputError(
            f"指纹维度不一致: {len(left)} 与 {len(right)}，无法计算相似度"
        )
    total = sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
    return math.sqrt(total)


def round_score(value: float) -> float:
    return round(float(value), 6)


@dataclass(frozen=True, slots=True)
class DesignRecord:
    design_id: str
    family_id: str
    first_available_at: str
    features: tuple[float, ...]
    fingerprint_id: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DesignRecord":
        design_id = str(payload.get("design_id", "")).strip()
        family_id = str(payload.get("family_id", "")).strip()
        if not design_id:
            raise StudyInputError("designs 中存在缺少 design_id 的记录")
        if not family_id:
            raise StudyInputError(f"设计 {design_id} 缺少 family_id")
        instant = parse_instant(payload.get("first_available_at"))
        features = coerce_features(payload.get("features"), f"设计 {design_id}")
        return cls(design_id, family_id, instant, features, fingerprint_vector(features))


@dataclass(frozen=True, slots=True)
class QueryRecord:
    query_id: str
    family_id: str
    at: str
    features: tuple[float, ...]
    fingerprint_id: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "QueryRecord":
        query_id = str(payload.get("query_id", "")).strip()
        family_id = str(payload.get("family_id", "")).strip()
        if not query_id:
            raise StudyInputError("queries 中存在缺少 query_id 的记录")
        if not family_id:
            raise StudyInputError(f"查询 {query_id} 缺少 family_id")
        instant = parse_instant(payload.get("at"))
        features = coerce_features(payload.get("features"), f"查询 {query_id}")
        return cls(query_id, family_id, instant, features, fingerprint_vector(features))


@dataclass(frozen=True, slots=True)
class JudgmentRecord:
    query_id: str
    design_id: str
    relevant: bool
    selected_by: str | None
    available_at: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "JudgmentRecord":
        query_id = str(payload.get("query_id", "")).strip()
        design_id = str(payload.get("design_id", "")).strip()
        if not query_id or not design_id:
            raise StudyInputError("judgments 记录必须同时包含 query_id 与 design_id")
        if not isinstance(payload.get("relevant"), bool):
            raise StudyInputError(
                f"裁决 {query_id}/{design_id} 的 relevant 必须是布尔值；"
                "未审候选应缺省而不是写成 false"
            )
        selected_by = payload.get("selected_by")
        if selected_by is not None:
            selected_by = str(selected_by)
        return cls(query_id, design_id, bool(payload["relevant"]), selected_by, parse_instant(payload.get("available_at")))


@dataclass(frozen=True, slots=True)
class CandidateView:
    """单条查询-候选视角，携带相似度、去重层级与标签状态。"""

    design_id: str
    family_id: str
    fingerprint_id: str
    distance: float
    similarity_score: float
    rank: int
    label_state: str  # relevant | not_relevant | unknown
    relevant: bool | None
    selected_by: str | None
    judgment_available_at: str | None
    exact_duplicate_rank: int | None
    family_duplicate_rank: int | None
    eligible: bool
    exclusion_reason: str | None


def similarity_from_distance(distance: float) -> float:
    """把非负距离映射到 (0,1] 的相似度，距离越近分数越高。"""
    return 1.0 / (1.0 + max(0.0, distance))
