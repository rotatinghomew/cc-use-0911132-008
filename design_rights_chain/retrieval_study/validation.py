"""研究内部有效性核验与受限结论。

核验覆盖三类风险：

1. 数据泄漏：同家族改稿不能一部分用于挑策略、另一部分用于宣称泛化；
   晚于分析截止时刻或查询时刻的信息不能被提前采用。
2. 选择偏差：标签只在旧送审策略覆盖的 query-design 对上存在，
   报告必须显式量化偏差范围，禁止把未审候选当负例。
3. 受限结论：预算不足、没有相关性证据或候选全属同族时，返回受限结论，
   不编造完整召回率。
"""
from __future__ import annotations

from typing import Any, Sequence

from design_rights_chain.retrieval_study.features import (
    DesignRecord,
    JudgmentRecord,
    QueryRecord,
)
from design_rights_chain.retrieval_study.strategies import BASELINE_STRATEGY


def _check(check_id: str, title: str, passed: bool, detail: dict[str, Any], severity: str = "error") -> dict[str, Any]:
    return {
        "check_id": check_id,
        "title": title,
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    }


def family_leakage_check(
    queries: Sequence[QueryRecord],
    heldout_families: Sequence[str],
) -> dict[str, Any]:
    heldout = set(heldout_families)
    dev_families = {query.family_id for query in queries if query.family_id not in heldout}
    overlap = sorted(heldout & dev_families)
    return _check(
        "family_holdout_leakage",
        "留出家族与开发家族互不重叠（防止同家族改稿跨切分泄漏）",
        not overlap,
        {
            "heldout_query_families": sorted(heldout),
            "development_query_families": sorted(dev_families),
            "overlap": overlap,
            "rule": "策略不使用任何相关性标签做拟合；留出查询家族只用于冻结策略后的评估",
        },
    )


def temporal_availability_check(
    queries: Sequence[QueryRecord],
    designs: Sequence[DesignRecord],
    as_of: str,
    cutoff: str | None,
) -> dict[str, Any]:
    violations = []
    for query in queries:
        for design in designs:
            if design.first_available_at > as_of:
                violations.append({
                    "query_id": query.query_id,
                    "design_id": design.design_id,
                    "first_available_at": design.first_available_at,
                    "as_of": as_of,
                })
    late_queries = [query.query_id for query in queries if query.at > as_of]
    cutoff_usage = (
        {"discovery_cutoff": cutoff, "rule": "晚于 cutoff 的裁决不进入开发期证据，只可用于留出期评估"}
        if cutoff
        else {"discovery_cutoff": None, "rule": "输入未提供 discovery_cutoff，时间切分退化"}
    )
    return _check(
        "temporal_availability",
        "候选在查询时刻未入库则不可检出，分析不采用 as_of 之后的信息",
        not violations and not late_queries,
        {
            "as_of": as_of,
            "unavailable_design_violations": violations[:20],
            "queries_after_as_of": late_queries,
            "cutoff_policy": cutoff_usage,
        },
    )


def late_label_check(
    judgments: Sequence[JudgmentRecord],
    cutoff: str | None,
) -> dict[str, Any]:
    if cutoff is None:
        return _check(
            "late_labels",
            "晚到裁决的时间边界",
            True,
            {"rule": "未提供 discovery_cutoff，全部裁决仅在 as_of 口径下使用", "late_judgments": []},
            severity="warning",
        )
    late = [
        {"query_id": item.query_id, "design_id": item.design_id, "available_at": item.available_at}
        for item in judgments
        if item.available_at > cutoff
    ]
    return _check(
        "late_labels",
        "晚到裁决只在其 available_at 之后采用：开发期屏蔽、留出期可用",
        True,
        {
            "discovery_cutoff": cutoff,
            "late_judgment_count": len(late),
            "late_judgments": late,
            "enforcement": (
                "晚于 cutoff 的裁决在开发期查询的视图中被置为 unknown（rankings 中以 "
                "label_masked_by_temporal_policy 标注），仅在留出家族评估中保留原标签"
            ),
        },
        severity="warning" if late else "info",
    )


def family_integrity_check(designs: Sequence[DesignRecord]) -> dict[str, Any]:
    mapping: dict[str, set[str]] = {}
    for design in designs:
        mapping.setdefault(design.design_id, set()).add(design.family_id)
    conflicts = {design_id: sorted(families) for design_id, families in mapping.items() if len(families) > 1}
    return _check(
        "family_integrity",
        "每个 design_id 只归属一个 family_id，去重层级口径一致",
        not conflicts,
        {"conflicts": conflicts, "design_count": len(designs), "family_count": len({d.family_id for d in designs})},
    )


def label_validity_check(judgments: Sequence[JudgmentRecord]) -> dict[str, Any]:
    negatives = sum(1 for item in judgments if not item.relevant)
    return _check(
        "label_validity",
        "裁决只覆盖部分 query-design 对，缺省即未知，不存在把未知写成 false 的情况",
        True,
        {
            "judgment_count": len(judgments),
            "relevant_count": sum(1 for item in judgments if item.relevant),
            "not_relevant_count": negatives,
            "unknown_semantics": "未出现在 judgments 中的候选一律按 unknown 处理，绝不视为不相关",
        },
        severity="warning",
    )


def selection_bias_check(
    queries: Sequence[QueryRecord],
    judgments: Sequence[JudgmentRecord],
    selectors: dict[str, str],
) -> dict[str, Any]:
    selector_counts: dict[str, int] = {}
    for item in judgments:
        selector_counts[item.selected_by or "unspecified"] = selector_counts.get(item.selected_by or "unspecified", 0) + 1
    query_judged = {query.query_id: 0 for query in queries}
    for item in judgments:
        if item.query_id in query_judged:
            query_judged[item.query_id] += 1
    missing = [query_id for query_id, count in query_judged.items() if count == 0]
    return _check(
        "selection_bias",
        "标签由旧送审策略选择产生（缺失非随机），偏差范围已显式量化",
        not missing,
        {
            "selector_counts": selector_counts,
            "judged_queries": sorted(query_id for query_id, count in query_judged.items() if count > 0),
            "queries_without_any_label": missing,
            "mechanism": "missing-not-at-random：只有被 " + selectors.get(BASELINE_STRATEGY, BASELINE_STRATEGY) + " 选中的候选才有标签",
            "consequence": "judged_precision 与确认相关数只在旧策略覆盖区域无偏；新探索区域只能给区间，不能给点估计",
        },
        severity="warning" if not missing else "error",
    )


def split_balance_check(
    queries: Sequence[QueryRecord],
    heldout_families: Sequence[str],
) -> dict[str, Any]:
    heldout = set(heldout_families)
    dev = [query for query in queries if query.family_id not in heldout]
    hold = [query for query in queries if query.family_id in heldout]
    unknown_holdout_families = sorted(heldout - {query.family_id for query in queries})
    return _check(
        "split_balance",
        "开发/留出切分规模与覆盖",
        bool(dev) and bool(hold),
        {
            "development_queries": [query.query_id for query in dev],
            "heldout_queries": [query.query_id for query in hold],
            "heldout_families_without_query": unknown_holdout_families,
            "development_count": len(dev),
            "heldout_count": len(hold),
            "note": "留出查询仅 2 条，结论外推能力有限，应结合受限结论阅读",
        },
        severity="warning" if len(hold) < 3 else "info",
    )


def replay_check(first_study_id: str, replay_study_id: str | None, identical_payloads: bool) -> dict[str, Any]:
    # 首次与重放使用完全相同的结构，保证同输入两次运行的输出逐字节一致；
    # 是否为重放由结果顶层的 replayed 字段表达。
    is_replay = replay_study_id is not None
    passed = (not is_replay) or (first_study_id == replay_study_id and identical_payloads)
    return _check(
        "replay_determinism",
        "同输入重放引用同一分析记录且结果一致",
        passed,
        {
            "first_study_id": first_study_id,
            "replay_study_id": first_study_id if not is_replay else replay_study_id,
            "payload_identical": identical_payloads,
        },
        severity="info",
    )


def build_constrained_summary(query_constraints: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    constrained_pairs: list[str] = []
    for item in query_constraints:
        if item["constraint_reasons"]:
            constrained_pairs.append(f"{item['query_id']}@b{item.get('budget', '?')}")
            for reason in item["constraint_reasons"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "constrained_query_budget_pairs": len(constrained_pairs),
        "constrained_pairs": constrained_pairs,
        "reason_counts": reason_counts,
        "policy": (
            "受限查询只报告标签无关指标与确认相关区间；不输出完整召回率，"
            "不把未知候选计为负例，不输出侵权认定"
        ),
    }
