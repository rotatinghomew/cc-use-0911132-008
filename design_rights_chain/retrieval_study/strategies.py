"""检索策略定义与去重层级。

包含两套策略：

* ``old_similarity`` 基线：直接按本地指纹相似度排序（距离越近越靠前），
  复刻旧系统"高分结果里前十条常是同一原稿裁切版本"的行为。
* ``family_diversified`` 改进策略：先在原稿家族层级去重，按家族最近邻距离
  决定家族轮询顺序，每个家族每轮只放行一个候选，保证有限名额先覆盖更多家族。

策略只使用查询时刻已入库（``first_available_at <= query.at``）的候选；
未入库作品不可被检出。排序键全部确定性，重放结果一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from design_rights_chain.retrieval_study.features import (
    CandidateView,
    DesignRecord,
    JudgmentRecord,
    QueryRecord,
    euclidean_distance,
    round_score,
    similarity_from_distance,
)

BASELINE_STRATEGY = "old_similarity"
DIVERSIFIED_STRATEGY = "family_diversified"
STRATEGY_ORDER = (BASELINE_STRATEGY, DIVERSIFIED_STRATEGY)

STRATEGY_DESCRIPTIONS = {
    BASELINE_STRATEGY: {
        "display_name": "旧相似度基线（无去重）",
        "ranking_rule": "按查询与候选数值指纹的欧氏距离升序排序，等价于相似度降序；同分时按 design_id 稳定打破平局",
        "dedup_level": "none",
        "label_free": True,
        "tunable_parameters": [],
    },
    DIVERSIFIED_STRATEGY: {
        "display_name": "家族轮询去重改进策略",
        "ranking_rule": (
            "先按 family_id 聚合候选，家族顺序由该家族最近邻候选与查询的距离决定；"
            "随后逐家族轮询，每轮每个家族只放行一个距离最近的候选，"
            "直到名额用尽或候选耗尽；同分时按 design_id 打破平局"
        ),
        "dedup_level": "family",
        "label_free": True,
        "tunable_parameters": [],
    },
}


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    design: DesignRecord
    distance: float

    @property
    def score(self) -> float:
        return similarity_from_distance(self.distance)


def score_candidates(query: QueryRecord, designs: Sequence[DesignRecord]) -> list[ScoredCandidate]:
    scored = [
        ScoredCandidate(design, euclidean_distance(query.features, design.features))
        for design in designs
    ]
    return sorted(scored, key=lambda item: (item.distance, item.design.design_id))


def eligible_designs(query: QueryRecord, designs: Sequence[DesignRecord]) -> list[DesignRecord]:
    """作品未入库前不可被检出：只保留查询时刻之前已可见的候选。"""
    return [design for design in designs if design.first_available_at <= query.at]


def rank_baseline(query: QueryRecord, designs: Sequence[DesignRecord]) -> list[ScoredCandidate]:
    return score_candidates(query, eligible_designs(query, designs))


def rank_diversified(query: QueryRecord, designs: Sequence[DesignRecord]) -> list[ScoredCandidate]:
    scored = score_candidates(query, eligible_designs(query, designs))
    grouped: dict[str, list[ScoredCandidate]] = {}
    for item in scored:
        grouped.setdefault(item.design.family_id, []).append(item)
    # 家族顺序：家族内最近邻距离升序，平局按 family_id。
    family_order = sorted(
        grouped,
        key=lambda family: (grouped[family][0].distance, family),
    )
    queues = {family: list(grouped[family]) for family in family_order}
    ordered: list[ScoredCandidate] = []
    total = len(scored)
    while len(ordered) < total:
        progressed = False
        for family in family_order:
            queue = queues[family]
            if queue:
                ordered.append(queue.pop(0))
                progressed = True
        if not progressed:
            break
    return ordered


RANKERS = {
    BASELINE_STRATEGY: rank_baseline,
    DIVERSIFIED_STRATEGY: rank_diversified,
}


def build_views(
    strategy_id: str,
    query: QueryRecord,
    ranked: Sequence[ScoredCandidate],
    judgments: dict[tuple[str, str], JudgmentRecord],
    all_designs: Sequence[DesignRecord],
) -> list[CandidateView]:
    """把排序结果展开为带标签状态与去重层级的候选视角。"""
    eligible_ids = {item.design.design_id for item in ranked}
    seen_fingerprints: dict[str, int] = {}
    seen_families: dict[str, int] = {}
    views: list[CandidateView] = []
    for position, item in enumerate(ranked, start=1):
        design = item.design
        judgment = judgments.get((query.query_id, design.design_id))
        if judgment is None:
            label_state = "unknown"
            relevant = None
            selected_by = None
            judgment_at = None
        else:
            label_state = "relevant" if judgment.relevant else "not_relevant"
            relevant = judgment.relevant
            selected_by = judgment.selected_by
            judgment_at = judgment.available_at
        exact_rank = seen_fingerprints.get(design.fingerprint_id)
        family_rank = seen_families.get(design.family_id)
        seen_fingerprints.setdefault(design.fingerprint_id, position)
        seen_families.setdefault(design.family_id, position)
        views.append(
            CandidateView(
                design_id=design.design_id,
                family_id=design.family_id,
                fingerprint_id=design.fingerprint_id,
                distance=round_score(item.distance),
                similarity_score=round_score(item.score),
                rank=position,
                label_state=label_state,
                relevant=relevant,
                selected_by=selected_by,
                judgment_available_at=judgment_at,
                exact_duplicate_rank=exact_rank,
                family_duplicate_rank=family_rank,
                eligible=True,
                exclusion_reason=None,
            )
        )
    # 查询时刻尚未入库的作品单列为排除项，保证"不可检出"可审计。
    excluded = [
        design
        for design in all_designs
        if design.design_id not in eligible_ids and design.first_available_at > query.at
    ]
    for design in sorted(excluded, key=lambda record: (record.first_available_at, record.design_id)):
        distance = euclidean_distance(query.features, design.features)
        views.append(
            CandidateView(
                design_id=design.design_id,
                family_id=design.family_id,
                fingerprint_id=design.fingerprint_id,
                distance=round_score(distance),
                similarity_score=round_score(similarity_from_distance(distance)),
                rank=0,
                label_state="unknown",
                relevant=None,
                selected_by=None,
                judgment_available_at=None,
                exact_duplicate_rank=None,
                family_duplicate_rank=None,
                eligible=False,
                exclusion_reason="not_available_at_query_time",
            )
        )
    return views
