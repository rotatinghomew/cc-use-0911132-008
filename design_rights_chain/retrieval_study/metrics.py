"""有限复核预算下的效果、重复占位、覆盖与不确定性度量。

核心口径（必须在报告中保持一致）：

* 未审候选（``label_state == 'unknown'``）是未知，不是负例，绝不参与
  精确率/召回率的分母惩罚，也不被当成不相关。
* 完整召回率无法计算：真实相关总量未知，因此 ``recall`` 一律为 ``None``，
  只给出"已确认相关数"与"确认相关的乐观上界"。
* 重复占位与家族覆盖不依赖标签，选择偏差存在时仍然成立，是本研究的主比较口径。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from design_rights_chain.retrieval_study.features import CandidateView
from design_rights_chain.retrieval_study.strategies import DIVERSIFIED_STRATEGY


@dataclass(frozen=True, slots=True)
class QueryMetric:
    budget: int
    eligible_candidates: int
    reviewed: int
    confirmed_relevant: int
    judged_in_review: int
    unknown_in_review: int
    not_relevant_in_review: int
    confirmed_relevant_lower_bound: int
    confirmed_relevant_upper_bound: int
    family_coverage: int
    eligible_family_total: int
    family_coverage_ratio: float | None
    exact_duplicate_slots: int
    family_duplicate_slots: int
    duplicate_slot_ratio: float
    unknown_slot_ratio: float
    judged_precision: float | None
    recall: None
    recall_note: str
    constrained: bool
    constraint_reasons: list[str]


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def evaluate_query(
    views: Sequence[CandidateView],
    budget: int,
    eligible_family_total: int,
) -> QueryMetric:
    eligible = [view for view in views if view.eligible]
    reviewed = eligible[: min(budget, len(eligible))]
    confirmed = sum(1 for view in reviewed if view.label_state == "relevant")
    judged = sum(1 for view in reviewed if view.label_state in ("relevant", "not_relevant"))
    unknown = sum(1 for view in reviewed if view.label_state == "unknown")
    not_relevant = sum(1 for view in reviewed if view.label_state == "not_relevant")
    exact_dup = sum(1 for view in reviewed if view.exact_duplicate_rank is not None)
    family_dup = sum(1 for view in reviewed if view.family_duplicate_rank is not None)
    families = {view.family_id for view in reviewed}

    reasons: list[str] = []
    if not reviewed:
        reasons.append("no_eligible_candidate")
    if judged == 0:
        # 送审名额里没有任何相关性证据：不能给精确率点估计，只能给区间。
        reasons.append("no_labeled_evidence_in_review")
    if confirmed == 0:
        # 没有已确认相关：有效发现下界为零，但未知候选可能仍相关。
        reasons.append("no_confirmed_relevance")
    if len({view.family_id for view in eligible}) <= 1:
        reasons.append("single_family_candidates")
    if budget > len(eligible):
        reasons.append("budget_exceeds_candidates")
    if unknown == len(reviewed) and reviewed:
        reasons.append("all_reviewed_candidates_unknown")

    return QueryMetric(
        budget=budget,
        eligible_candidates=len(eligible),
        reviewed=len(reviewed),
        confirmed_relevant=confirmed,
        judged_in_review=judged,
        unknown_in_review=unknown,
        not_relevant_in_review=not_relevant,
        # 下界只数已确认相关；上界假设所有未知候选最终都相关。
        confirmed_relevant_lower_bound=confirmed,
        confirmed_relevant_upper_bound=confirmed + unknown,
        family_coverage=len(families),
        eligible_family_total=eligible_family_total,
        family_coverage_ratio=_safe_ratio(len(families), eligible_family_total),
        exact_duplicate_slots=exact_dup,
        family_duplicate_slots=family_dup,
        duplicate_slot_ratio=round(family_dup / len(reviewed), 6) if reviewed else 0.0,
        unknown_slot_ratio=round(unknown / len(reviewed), 6) if reviewed else 0.0,
        judged_precision=_safe_ratio(confirmed, judged),
        recall=None,
        recall_note=(
            "真实相关总量未知：未审候选是未知而非负例，不能构造完整召回率；"
            "仅报告已确认相关数及其在未知全部相关假设下的上界"
        ),
        constrained=bool(reasons),
        constraint_reasons=reasons,
    )


def aggregate(metrics: Sequence[QueryMetric]) -> dict[str, Any]:
    """对一组同预算查询做微平均与宏平均；空切分也返回完整结构，避免下游猜字段。"""
    if not metrics:
        return {
            "query_count": 0,
            "note": "该切分下没有可评估查询（例如留出家族在本输入中没有查询）",
            "micro": {
                "reviewed_slots": 0,
                "confirmed_relevant": 0,
                "confirmed_relevant_lower_bound": 0,
                "confirmed_relevant_upper_bound": 0,
                "judged_slots": 0,
                "unknown_slots": 0,
                "family_duplicate_slots": 0,
                "duplicate_slot_ratio": None,
                "unknown_slot_ratio": None,
                "judged_precision": None,
                "recall": None,
            },
            "macro": {
                "confirmed_relevant_per_query": None,
                "family_duplicate_slots_per_query": None,
                "family_coverage_ratio": None,
                "unknown_slot_ratio": None,
                "judged_precision": None,
                "recall": None,
            },
            "constrained_query_count": 0,
        }
    reviewed_total = sum(item.reviewed for item in metrics)
    confirmed_total = sum(item.confirmed_relevant for item in metrics)
    judged_total = sum(item.judged_in_review for item in metrics)
    unknown_total = sum(item.unknown_in_review for item in metrics)
    family_dup_total = sum(item.family_duplicate_slots for item in metrics)
    coverage_values = [item.family_coverage_ratio for item in metrics if item.family_coverage_ratio is not None]
    return {
        "query_count": len(metrics),
        "budget": metrics[0].budget,
        "micro": {
            "reviewed_slots": reviewed_total,
            "confirmed_relevant": confirmed_total,
            "confirmed_relevant_lower_bound": confirmed_total,
            "confirmed_relevant_upper_bound": confirmed_total + unknown_total,
            "judged_slots": judged_total,
            "unknown_slots": unknown_total,
            "family_duplicate_slots": family_dup_total,
            "duplicate_slot_ratio": round(family_dup_total / reviewed_total, 6) if reviewed_total else 0.0,
            "unknown_slot_ratio": round(unknown_total / reviewed_total, 6) if reviewed_total else 0.0,
            "judged_precision": round(confirmed_total / judged_total, 6) if judged_total else None,
            "recall": None,
        },
        "macro": {
            "confirmed_relevant_per_query": round(confirmed_total / len(metrics), 6),
            "family_duplicate_slots_per_query": round(family_dup_total / len(metrics), 6),
            "family_coverage_ratio": round(sum(coverage_values) / len(coverage_values), 6) if coverage_values else None,
            "unknown_slot_ratio": round(sum(item.unknown_slot_ratio for item in metrics) / len(metrics), 6),
            "judged_precision": round(
                sum(item.judged_precision for item in metrics if item.judged_precision is not None)
                / max(1, sum(1 for item in metrics if item.judged_precision is not None)),
                6,
            )
            if any(item.judged_precision is not None for item in metrics)
            else None,
            "recall": None,
        },
        "constrained_query_count": sum(1 for item in metrics if item.constrained),
    }


def paired_comparison(
    baseline_views: Sequence[CandidateView],
    diversified_views: Sequence[CandidateView],
    budget: int,
) -> dict[str, Any]:
    """只在两套策略都可观测的部分做配对比较，显式暴露选择偏差。"""
    base = {view.design_id: view for view in baseline_views if view.eligible}
    diver = {view.design_id: view for view in diversified_views if view.eligible}
    base_top = [view for view in baseline_views if view.eligible][:budget]
    diver_top = [view for view in diversified_views if view.eligible][:budget]
    base_ids = {view.design_id for view in base_top}
    diver_ids = {view.design_id for view in diver_top}
    shared = base_ids & diver_ids
    baseline_only = base_ids - diver_ids
    diversified_only = diver_ids - base_ids

    def label_summary(ids: set[str], pool: dict[str, CandidateView]) -> dict[str, Any]:
        relevant = sum(1 for did in ids if pool[did].label_state == "relevant")
        judged = sum(1 for did in ids if pool[did].label_state in ("relevant", "not_relevant"))
        return {
            "candidate_count": len(ids),
            "confirmed_relevant": relevant,
            "judged": judged,
            "unknown": len(ids) - judged,
        }

    return {
        "budget": budget,
        "shared_slots": len(shared),
        "baseline_only": label_summary(baseline_only, base),
        "diversified_only": label_summary(diversified_only, diver),
        "bias_note": (
            "标签只覆盖旧策略送审的候选：改进策略独有候选若从未被旧策略送审，"
            "其标签状态为未知。两策略的确认相关差值是有偏下界，不能当作改进策略的真实增益；"
            "标签无关的家族覆盖与重复占位比较不受该偏差影响。"
        ),
        "fair_claim": (
            "可成立的比较仅限：共享候选上的排序一致性、家族覆盖、重复占位；"
            "改进策略独有候选的相关性需后续主动送审才能验证"
        ),
        "diversified_strategy_id": DIVERSIFIED_STRATEGY,
    }


def metric_to_dict(metric: QueryMetric) -> dict[str, Any]:
    return asdict(metric)
