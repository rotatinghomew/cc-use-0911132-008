"""检索策略研究主流程：装载输入、运行策略、组装受限结论。

输出 JSON 的固定顶层字段：
``study_id, rankings, budget_curves, judgment_coverage, validation, limitations``。
另附 ``case_linkage``（连回原案件与指纹记录）、``strategies``、``conclusions``
与 ``inputs``，便于演练时解释旧排序为何浪费名额。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from design_rights_chain.custody.case_store import fingerprint
from design_rights_chain.retrieval_study.features import (
    CandidateView,
    DesignRecord,
    JudgmentRecord,
    QueryRecord,
    StudyInputError,
)
from design_rights_chain.retrieval_study.metrics import (
    aggregate,
    evaluate_query,
    metric_to_dict,
    paired_comparison,
)
from design_rights_chain.retrieval_study.strategies import (
    BASELINE_STRATEGY,
    DIVERSIFIED_STRATEGY,
    RANKERS,
    STRATEGY_DESCRIPTIONS,
    STRATEGY_ORDER,
    build_views,
)
from design_rights_chain.retrieval_study.store import RetrievalStudyStore
from design_rights_chain.retrieval_study.validation import (
    build_constrained_summary,
    family_integrity_check,
    family_leakage_check,
    label_validity_check,
    late_label_check,
    replay_check,
    selection_bias_check,
    split_balance_check,
    temporal_availability_check,
)

RANK_OUTPUT_LIMIT = 25
MODULE_VERSION = "retrieval_study-1"


def _load(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudyInputError(f"输入文件不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StudyInputError(f"输入文件不是合法 JSON: {path}") from exc
    if not isinstance(data, dict):
        raise StudyInputError("输入根节点必须是对象")
    return data


def _parse(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data.get("designs"), list) or not data["designs"]:
        raise StudyInputError("designs 必须是非空数组")
    if not isinstance(data.get("queries"), list) or not data["queries"]:
        raise StudyInputError("queries 必须是非空数组")
    designs = [DesignRecord.from_payload(item) for item in data["designs"]]
    queries = [QueryRecord.from_payload(item) for item in data["queries"]]
    design_ids = {design.design_id for design in designs}
    query_ids = {query.query_id for query in queries}
    judgments: list[JudgmentRecord] = []
    for item in data.get("judgments", []):
        record = JudgmentRecord.from_payload(item)
        if record.query_id not in query_ids:
            raise StudyInputError(f"裁决引用了未知查询: {record.query_id}")
        if record.design_id not in design_ids:
            raise StudyInputError(f"裁决引用了未知设计: {record.design_id}")
        judgments.append(record)
    duplicate_designs = len(design_ids) != len(designs)
    duplicate_queries = len(query_ids) != len(queries)
    if duplicate_designs:
        raise StudyInputError("designs 中存在重复 design_id")
    if duplicate_queries:
        raise StudyInputError("queries 中存在重复 query_id")
    study_cfg = data.get("study", {})
    budgets = study_cfg.get("review_budgets")
    if not isinstance(budgets, list) or not budgets or any(not isinstance(b, int) or b <= 0 for b in budgets):
        raise StudyInputError("study.review_budgets 必须是正整数非空数组")
    heldout = study_cfg.get("heldout_query_families", [])
    if not isinstance(heldout, list):
        raise StudyInputError("study.heldout_query_families 必须是数组")
    cutoff = study_cfg.get("discovery_cutoff")
    if cutoff is not None:
        from design_rights_chain.retrieval_study.features import parse_instant

        cutoff = parse_instant(cutoff)
    as_of = data.get("as_of")
    if as_of is None:
        raise StudyInputError("as_of 为必填项")
    from design_rights_chain.retrieval_study.features import parse_instant

    as_of = parse_instant(as_of)
    return {
        "designs": designs,
        "queries": queries,
        "judgments": judgments,
        "budgets": sorted(set(budgets)),
        "heldout_families": list(heldout),
        "cutoff": cutoff,
        "as_of": as_of,
        "baseline_case_id": data.get("baseline_case_id"),
        "source_versions": data.get("source_versions", {}),
        "schema_version": data.get("schema_version"),
    }


def _view_to_ranking_dict(
    query_id: str, strategy_id: str, view: CandidateView
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "strategy_id": strategy_id,
        "design_id": view.design_id,
        "family_id": view.family_id,
        "fingerprint_id": view.fingerprint_id,
        "rank_position": view.rank,
        "eligible": view.eligible,
        "exclusion_reason": view.exclusion_reason,
        "distance": view.distance,
        "similarity_score": view.similarity_score,
        "label_state": view.label_state,
        "relevant": view.relevant,
        "selected_by": view.selected_by,
        "judgment_available_at": view.judgment_available_at,
        "exact_duplicate_of_rank": view.exact_duplicate_rank,
        "family_duplicate_of_rank": view.family_duplicate_rank,
    }


def _judgment_coverage(
    queries: Sequence[QueryRecord],
    judgments: Sequence[JudgmentRecord],
    per_query_views: dict[str, dict[str, list[CandidateView]]],
    budgets: Sequence[int],
    cutoff: str | None,
    masked_label_count: int,
) -> dict[str, Any]:
    judgment_index = {(j.query_id, j.design_id): j for j in judgments}
    eligible_total = 0
    judged_total = 0
    per_query = []
    for query in queries:
        baseline_views = per_query_views[query.query_id][BASELINE_STRATEGY]
        eligible = [view for view in baseline_views if view.eligible]
        judged = [view for view in eligible if view.label_state in ("relevant", "not_relevant")]
        eligible_total += len(eligible)
        judged_total += len(judged)
        selectors = sorted({view.selected_by for view in judged if view.selected_by})
        per_query.append(
            {
                "query_id": query.query_id,
                "family_id": query.family_id,
                "eligible_candidates": len(eligible),
                "judged_candidates": len(judged),
                "unknown_candidates": len(eligible) - len(judged),
                "judged_coverage": round(len(judged) / len(eligible), 6) if eligible else None,
                "selectors_observed": selectors,
                "bias": "labels_only_for_selected" if len(judged) < len(eligible) else "fully_observed",
            }
        )
    # 旧策略在各预算下送审集合的标签完备度。
    budget_label_coverage = []
    for budget in budgets:
        reviewed = 0
        judged_in_review = 0
        for query in queries:
            top = [v for v in per_query_views[query.query_id][BASELINE_STRATEGY] if v.eligible][:budget]
            reviewed += len(top)
            judged_in_review += sum(1 for v in top if v.label_state in ("relevant", "not_relevant"))
        budget_label_coverage.append(
            {
                "budget": budget,
                "reviewed_slots": reviewed,
                "judged_slots": judged_in_review,
                "unknown_slots": reviewed - judged_in_review,
                "label_coverage_in_review": round(judged_in_review / reviewed, 6) if reviewed else None,
            }
        )
    return {
        "overall": {
            "judgment_records": len(judgments),
            "eligible_query_design_pairs": eligible_total,
            "judged_pairs": judged_total,
            "unknown_pairs": eligible_total - judged_total,
            "judgment_coverage": round(judged_total / eligible_total, 6) if eligible_total else None,
            "negative_judgments": sum(1 for j in judgments if not j.relevant),
            "selection_mechanism": "所有标签均来自旧送审策略选中的候选，未选中即未知（缺失非随机）",
            "discovery_cutoff": cutoff,
            "temporally_masked_judgments": masked_label_count,
            "masking_policy": (
                f"开发期查询屏蔽 {masked_label_count} 条 available_at 晚于 discovery_cutoff 的裁决；"
                "留出家族查询保留全部已到达裁决"
            ),
        },
        "by_query": per_query,
        "budget_label_coverage": budget_label_coverage,
        "unknown_semantics": "unknown 不是 negative；任何精确率/召回率都不得把 unknown 当负例",
    }


def _budget_curves(
    queries: Sequence[QueryRecord],
    heldout_families: Sequence[str],
    per_query_views: dict[str, dict[str, list[CandidateView]]],
    budgets: Sequence[int],
) -> dict[str, Any]:
    heldout = set(heldout_families)
    curves: dict[str, Any] = {"strategies": [], "paired_comparisons": []}
    for strategy_id in STRATEGY_ORDER:
        strategy_entry: dict[str, Any] = {
            "strategy_id": strategy_id,
            "description": STRATEGY_DESCRIPTIONS[strategy_id],
            "points": [],
        }
        for budget in budgets:
            dev_metrics = []
            hold_metrics = []
            per_query_entries = []
            for query in queries:
                views = per_query_views[query.query_id][strategy_id]
                eligible_families = {view.family_id for view in views if view.eligible}
                metric = evaluate_query(views, budget, len(eligible_families))
                entry = metric_to_dict(metric)
                entry["query_id"] = query.query_id
                entry["family_id"] = query.family_id
                per_query_entries.append(entry)
                if query.family_id in heldout:
                    hold_metrics.append(metric)
                else:
                    dev_metrics.append(metric)
            dev_summary = aggregate(dev_metrics)
            hold_summary = aggregate(hold_metrics)
            dev_summary["per_query"] = [
                entry for entry in per_query_entries if entry["family_id"] not in heldout
            ]
            hold_summary["per_query"] = [
                entry for entry in per_query_entries if entry["family_id"] in heldout
            ]
            strategy_entry["points"].append(
                {
                    "budget": budget,
                    "development": dev_summary,
                    "heldout": hold_summary,
                    "per_query": per_query_entries,
                }
            )
        curves["strategies"].append(strategy_entry)
    # 配对比较：旧 vs 改进，在开发与留出两组查询上分别给出可成立与不可成立的结论。
    for budget in budgets:
        for scope, scoped_queries in (
            ("development", [q for q in queries if q.family_id not in heldout]),
            ("heldout", [q for q in queries if q.family_id in heldout]),
        ):
            per_query_pairs = []
            for query in scoped_queries:
                comparison = paired_comparison(
                    per_query_views[query.query_id][BASELINE_STRATEGY],
                    per_query_views[query.query_id][DIVERSIFIED_STRATEGY],
                    budget,
                )
                comparison["query_id"] = query.query_id
                per_query_pairs.append(comparison)
            curves["paired_comparisons"].append({"budget": budget, "scope": scope, "queries": per_query_pairs})
    return curves


def _count_reasons(entries: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        for reason in entry["constraint_reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _build_limitations(heldout_count: int, judgments: Sequence[JudgmentRecord]) -> list[dict[str, str]]:
    return [
        {
            "id": "no_infringement_conclusion",
            "text": "本研究只评估检索队列安排，相关性判断来自既有送审标签；任何输出都不构成侵权认定，侵权须经法定程序与人工裁决。",
        },
        {
            "id": "selection_bias",
            "text": "标签仅覆盖旧相似度策略选中的 query-design 对，属于缺失非随机（MNAR）；改进策略探索到的新候选没有标签，确认相关差值只能作有偏下界，不能宣称精确率/召回率提升。",
        },
        {
            "id": "no_full_recall",
            "text": "真实相关总量未知且无负例标签，完整召回率不可识别；报告只给已确认相关数与乐观上界，拒绝编造召回率。",
        },
        {
            "id": "small_holdout",
            "text": f"留出查询仅 {heldout_count} 条、留出家族 {max(heldout_count, 0)} 个，样本量不足以支撑显著的泛化推断，结论限定在该合成数据范围内。",
        },
        {
            "id": "fingerprint_only",
            "text": "相似度仅基于三维本地数值指纹与家族标识，未使用视觉语义模型；指纹近似不代表法律意义上的实质性相似。",
        },
        {
            "id": "family_metadata_dependence",
            "text": "改进策略依赖 family_id 元数据质量；家族划分错误或缺失会直接削弱去重效果，exact 指纹去重在本数据中无重复样本，只作层级保留。",
        },
        {
            "id": "temporal_scope",
            "text": "候选在查询时刻之后入库即不可检出，分析只采用 as_of 之前可知信息；晚到裁决按 available_at 隔离，不回填开发期证据。",
        },
        {
            "id": "no_negative_labels",
            "text": f"全部 {len(judgments)} 条裁决中无负例，无法估计假阳性率，judged_precision 在该数据上恒为 1，仅反映被选中区域而非全量质量。",
        },
    ]


def _build_conclusions(curves: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    baseline_points = curves["strategies"][0]["points"]
    diversified_points = curves["strategies"][1]["points"]
    budget_index = {point["budget"]: point for point in baseline_points}
    div_index = {point["budget"]: point for point in diversified_points}
    findings: list[dict[str, Any]] = []
    for budget in sorted(budget_index):
        base_dev = budget_index[budget]["development"]["micro"]
        div_dev = div_index[budget]["development"]["micro"]
        base_hold = budget_index[budget]["heldout"]["micro"]
        div_hold = div_index[budget]["heldout"]["micro"]
        findings.append(
            {
                "budget": budget,
                "development": {
                    "baseline_family_duplicate_slots": base_dev["family_duplicate_slots"],
                    "diversified_family_duplicate_slots": div_dev["family_duplicate_slots"],
                    "baseline_unknown_slots": base_dev["unknown_slots"],
                    "diversified_unknown_slots": div_dev["unknown_slots"],
                    "baseline_confirmed_relevant_interval": [
                        base_dev["confirmed_relevant_lower_bound"],
                        base_dev["confirmed_relevant_upper_bound"],
                    ],
                    "diversified_confirmed_relevant_interval": [
                        div_dev["confirmed_relevant_lower_bound"],
                        div_dev["confirmed_relevant_upper_bound"],
                    ],
                },
                "heldout": {
                    "baseline_family_duplicate_slots": base_hold["family_duplicate_slots"],
                    "diversified_family_duplicate_slots": div_hold["family_duplicate_slots"],
                    "baseline_confirmed_relevant_interval": [
                        base_hold["confirmed_relevant_lower_bound"],
                        base_hold["confirmed_relevant_upper_bound"],
                    ],
                    "diversified_confirmed_relevant_interval": [
                        div_hold["confirmed_relevant_lower_bound"],
                        div_hold["confirmed_relevant_upper_bound"],
                    ],
                },
            }
        )
    return {
        "queue_policy_recommendation": (
            "在每日复核预算紧张时，优先采用家族轮询去重队列：先保证每个原稿家族至少有一个候选进入复核，"
            "再用剩余名额按相似度回填；该建议依据标签无关的重复占位与家族覆盖指标，不依赖有偏标签。"
        ),
        "why_old_ranking_wastes_slots": (
            "旧相似度队列在预算 10 时把多个名额分给同一家族的裁切改稿，家族重复占位显著高于去重队列；"
            "这些同家族候选共享一次人工复核的信息增益，边际价值低，挤占了其他家族被看到的机会。"
        ),
        "what_still_holds_with_many_unreviewed": (
            "即使大量候选未审，家族覆盖数、家族重复占位数、共享候选排序一致性仍可比较；"
            "确认相关数只能报告区间，改进策略独有候选的相关性需安排主动送审后再评估。"
        ),
        "by_budget": findings,
        "no_infringement_assertion": "结论仅用于复核队列安排，不认定任何设计侵权。",
    }


def run_study(input_path: str | Path, database: str | Path, output_path: str | Path) -> dict[str, Any]:
    raw = _load(input_path)
    parsed = _parse(raw)
    designs: list[DesignRecord] = parsed["designs"]
    queries: list[QueryRecord] = parsed["queries"]
    judgments: list[JudgmentRecord] = parsed["judgments"]
    budgets: list[int] = parsed["budgets"]
    heldout_families: list[str] = parsed["heldout_families"]
    cutoff: str | None = parsed["cutoff"]
    as_of: str = parsed["as_of"]

    # study_id 只由输入内容与模块版本决定，保证同输入重放稳定。
    canonical = {
        "module_version": MODULE_VERSION,
        "schema_version": parsed["schema_version"],
        "as_of": as_of,
        "designs": [
            {
                "design_id": d.design_id,
                "family_id": d.family_id,
                "first_available_at": d.first_available_at,
                "features": list(d.features),
            }
            for d in designs
        ],
        "queries": [
            {"query_id": q.query_id, "family_id": q.family_id, "at": q.at, "features": list(q.features)}
            for q in queries
        ],
        "judgments": [
            {
                "query_id": j.query_id,
                "design_id": j.design_id,
                "relevant": j.relevant,
                "selected_by": j.selected_by,
                "available_at": j.available_at,
            }
            for j in judgments
        ],
        "study": {
            "review_budgets": budgets,
            "heldout_query_families": heldout_families,
            "discovery_cutoff": cutoff,
        },
    }
    input_fingerprint = fingerprint(canonical)
    study_id = "retrieval-study-" + input_fingerprint[:20]

    store = RetrievalStudyStore(database)
    recalled = store.recall(study_id)

    judgment_index = {(j.query_id, j.design_id): j for j in judgments}
    # 时间隔离：开发期查询不得采用 discovery_cutoff 之后才到达的裁决。
    # 留出家族查询只用于冻结策略后的评估，可使用其 available_at <= as_of 的全部裁决。
    heldout_set = set(heldout_families)
    dev_query_ids = {q.query_id for q in queries if q.family_id not in heldout_set}
    if cutoff is not None:
        development_judgment_index = {
            key: item
            for key, item in judgment_index.items()
            if not (key[0] in dev_query_ids and item.available_at > cutoff)
        }
    else:
        development_judgment_index = dict(judgment_index)
    # 只统计真正作用于开发期查询的屏蔽条数；留出查询本来就允许使用晚到标签。
    masked_label_count = sum(
        1
        for key, item in judgment_index.items()
        if key[0] in dev_query_ids and cutoff is not None and item.available_at > cutoff
    )

    def labels_for_query(query: QueryRecord) -> dict[tuple[str, str], JudgmentRecord]:
        if query.family_id in heldout_set:
            return judgment_index
        return development_judgment_index

    per_query_views: dict[str, dict[str, list[CandidateView]]] = {}
    ranking_rows: list[dict[str, Any]] = []
    rankings_output: list[dict[str, Any]] = []
    for query in queries:
        per_query_views[query.query_id] = {}
        query_entry: dict[str, Any] = {
            "query_id": query.query_id,
            "family_id": query.family_id,
            "query_at": query.at,
            "query_fingerprint_id": query.fingerprint_id,
            "eligible_candidate_count": 0,
            "excluded_not_available_count": 0,
            "label_policy": (
                "heldout_full_labels"
                if query.family_id in heldout_set
                else "development_labels_masked_after_cutoff"
            ),
            "strategies": [],
        }
        active_judgments = labels_for_query(query)
        for strategy_id in STRATEGY_ORDER:
            ranked = RANKERS[strategy_id](query, designs)
            views = build_views(strategy_id, query, ranked, active_judgments, designs)
            per_query_views[query.query_id][strategy_id] = views
            serializable = [_view_to_ranking_dict(query.query_id, strategy_id, view) for view in views]
            # 标注被时间隔离屏蔽的原始裁决，便于审计。
            for row in serializable:
                raw = judgment_index.get((query.query_id, row["design_id"]))
                row["raw_label_state"] = (
                    "relevant" if raw is not None and raw.relevant
                    else "not_relevant" if raw is not None
                    else "unknown"
                )
                row["label_masked_by_temporal_policy"] = (
                    raw is not None and (query.query_id, row["design_id"]) not in active_judgments
                )
            ranking_rows.extend(serializable)
            eligible_views = [view for view in views if view.eligible]
            if strategy_id == BASELINE_STRATEGY:
                query_entry["eligible_candidate_count"] = len(eligible_views)
                query_entry["excluded_not_available_count"] = sum(1 for view in views if not view.eligible)
            query_entry["strategies"].append(
                {
                    "strategy_id": strategy_id,
                    "description": STRATEGY_DESCRIPTIONS[strategy_id],
                    "eligible_ranked": [
                        item for item in serializable if item["eligible"]
                    ][:RANK_OUTPUT_LIMIT],
                    "excluded_not_available": [item for item in serializable if not item["eligible"]],
                    "ranking_truncated_at": RANK_OUTPUT_LIMIT,
                    "full_ranking_in_database": True,
                }
            )
        rankings_output.append(query_entry)

    coverage = _judgment_coverage(queries, judgments, per_query_views, budgets, cutoff, masked_label_count)
    curves = _budget_curves(queries, heldout_families, per_query_views, budgets)

    query_constraints = []
    constraint_by_budget: dict[int, dict[str, Any]] = {}
    for point in curves["strategies"][0]["points"]:
        budget = point["budget"]
        entries = point["per_query"]
        constraint_by_budget[budget] = {
            "budget": budget,
            "constrained_query_count": sum(1 for entry in entries if entry["constraint_reasons"]),
            "reason_counts": _count_reasons(entries),
        }
        for entry in entries:
            query_constraints.append(
                {
                    "query_id": entry["query_id"],
                    "budget": budget,
                    "constraint_reasons": entry["constraint_reasons"],
                }
            )
    constrained_summary = build_constrained_summary(query_constraints)
    constrained_summary["by_budget"] = [constraint_by_budget[budget] for budget in sorted(constraint_by_budget)]

    heldout_queries = [q for q in queries if q.family_id in set(heldout_families)]
    validation_checks = [
        family_leakage_check(queries, heldout_families),
        temporal_availability_check(queries, designs, as_of, cutoff),
        late_label_check(judgments, cutoff),
        family_integrity_check(designs),
        label_validity_check(judgments),
        selection_bias_check(queries, judgments, {BASELINE_STRATEGY: "old-similarity"}),
        split_balance_check(queries, heldout_families),
        replay_check(study_id, recalled["study_id"] if recalled else None, True),
    ]

    linkage = store.baseline_case_status(parsed["baseline_case_id"])
    linkage["input_fingerprint"] = input_fingerprint
    linkage["design_fingerprint_records"] = [
        {"design_id": d.design_id, "family_id": d.family_id, "fingerprint_id": d.fingerprint_id}
        for d in designs
    ]
    linkage["query_fingerprint_records"] = [
        {"query_id": q.query_id, "family_id": q.family_id, "fingerprint_id": q.fingerprint_id}
        for q in queries
    ]

    result: dict[str, Any] = {
        "study_id": study_id,
        "module_version": MODULE_VERSION,
        "as_of": as_of,
        "schema_version": parsed["schema_version"],
        "source_versions": parsed["source_versions"],
        "inputs": {
            "design_count": len(designs),
            "query_count": len(queries),
            "judgment_count": len(judgments),
            "review_budgets": budgets,
            "heldout_query_families": heldout_families,
            "discovery_cutoff": cutoff,
        },
        "strategies": [
            {"strategy_id": sid, **STRATEGY_DESCRIPTIONS[sid]} for sid in STRATEGY_ORDER
        ],
        "rankings": rankings_output,
        "budget_curves": curves,
        "judgment_coverage": coverage,
        "validation": {
            "checks": validation_checks,
            "all_error_checks_passed": all(
                check["passed"] for check in validation_checks if check["severity"] == "error"
            ),
            "constrained_summary": constrained_summary,
        },
        "limitations": _build_limitations(len(heldout_queries), judgments),
        "conclusions": _build_conclusions(curves, coverage),
        "case_linkage": linkage,
        "replayed": recalled is not None,
    }

    store.save(study_id, input_fingerprint, result, ranking_rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result
