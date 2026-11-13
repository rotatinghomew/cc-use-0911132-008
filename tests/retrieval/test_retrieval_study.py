from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from design_rights_chain.retrieval_study import StudyInputError, run_study
from design_rights_chain.retrieval_study.features import (
    CandidateView,
    DesignRecord,
    QueryRecord,
    euclidean_distance,
)
from design_rights_chain.retrieval_study.strategies import (
    BASELINE_STRATEGY,
    DIVERSIFIED_STRATEGY,
    RANKERS,
    build_views,
    rank_diversified,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "analysis" / "input.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def design(design_id: str, family: str, available: str, features: list[float]) -> dict:
    return {
        "design_id": design_id,
        "family_id": family,
        "first_available_at": available,
        "features": features,
    }


def query(query_id: str, family: str, at: str, features: list[float]) -> dict:
    return {"query_id": query_id, "family_id": family, "at": at, "features": features}


def judgment(query_id: str, design_id: str, relevant: bool, at: str, selected_by: str = "old-similarity") -> dict:
    return {
        "query_id": query_id,
        "design_id": design_id,
        "relevant": relevant,
        "selected_by": selected_by,
        "available_at": at,
    }


def study(budgets: list[int], heldout: list[str], cutoff: str = "2026-09-25T00:00:00+00:00") -> dict:
    return {
        "review_budgets": budgets,
        "heldout_query_families": heldout,
        "discovery_cutoff": cutoff,
    }


class RetrievalStudyContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.input_path = self.workspace / "input.json"
        self.db_path = self.workspace / "study.sqlite3"
        self.out_path = self.workspace / "out.json"

    def run_payload(self, payload: dict) -> dict:
        self.input_path.write_text(json.dumps(payload), encoding="utf-8")
        return run_study(self.input_path, self.db_path, self.out_path)

    def base_payload(self) -> dict:
        payload = load_fixture()
        return payload

    def test_required_top_level_fields_and_replay(self):
        payload = self.base_payload()
        first = self.run_payload(payload)
        for field in ("study_id", "rankings", "budget_curves", "judgment_coverage", "validation", "limitations"):
            self.assertIn(field, first)
        # 同输入重放：同一 study_id、replayed 标记翻转、业务结果一致。
        second = self.run_payload(payload)
        self.assertEqual(first["study_id"], second["study_id"])
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        business_fields = (
            "study_id", "rankings", "budget_curves", "judgment_coverage", "limitations", "conclusions"
        )
        for field in business_fields:
            self.assertEqual(
                json.dumps(first[field], sort_keys=True, ensure_ascii=False),
                json.dumps(second[field], sort_keys=True, ensure_ascii=False),
                f"重放后字段 {field} 不一致",
            )
        replay_checks = [
            next(c for c in result["validation"]["checks"] if c["check_id"] == "replay_determinism")
            for result in (first, second)
        ]
        self.assertTrue(all(check["passed"] for check in replay_checks))
        self.assertEqual(
            replay_checks[0]["detail"]["first_study_id"],
            replay_checks[1]["detail"]["replay_study_id"],
        )
        self.assertEqual(replay_checks[0]["detail"], replay_checks[1]["detail"])

    def test_persists_to_independent_tables_without_touching_case_flow(self):
        from design_rights_chain.service import ApplicationService

        payload = self.base_payload()
        # 先在同库写入一条案件流记录，验证研究模块只读不写旧表。
        service = ApplicationService(self.db_path)
        service.execute({"case_id": payload["baseline_case_id"]}, request_key="seed-case")
        with sqlite3.connect(self.db_path) as connection:
            runs_before = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        self.run_payload(payload)
        self.run_payload(payload)
        with sqlite3.connect(self.db_path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("retrieval_studies", tables)
            self.assertIn("retrieval_rankings", tables)
            self.assertIn("baseline_case_links", tables)
            self.assertIn("runs", tables)
            runs_after = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            ranking_rows = connection.execute(
                "SELECT COUNT(*) FROM retrieval_rankings WHERE study_id=?",
                (json.loads(self.out_path.read_text())["study_id"],),
            ).fetchone()[0]
        self.assertEqual(runs_before, runs_after)
        self.assertGreater(ranking_rows, 0)

    def test_baseline_case_linkage_reflects_missing_case(self):
        payload = self.base_payload()
        payload["baseline_case_id"] = "does-not-exist"
        result = self.run_payload(payload)
        self.assertFalse(result["case_linkage"]["case_present"])
        self.assertIsNone(result["case_linkage"]["linked_run_id"])
        self.assertEqual(result["case_linkage"]["baseline_case_id"], "does-not-exist")

    def test_unavailable_designs_are_excluded_not_ranked(self):
        payload = self.base_payload()
        result = self.run_payload(payload)
        q0 = next(item for item in result["rankings"] if item["query_id"] == "query-0")
        # query-0 在 2026-08-26，只有 26 个作品已入库，其余必须被排除。
        self.assertEqual(q0["eligible_candidate_count"], 26)
        self.assertEqual(q0["excluded_not_available_count"], 22)
        for strategy in q0["strategies"]:
            for row in strategy["eligible_ranked"]:
                self.assertTrue(row["eligible"])
                self.assertGreater(row["rank_position"], 0)
            for row in strategy["excluded_not_available"]:
                self.assertFalse(row["eligible"])
                self.assertEqual(row["exclusion_reason"], "not_available_at_query_time")
                self.assertEqual(row["rank_position"], 0)

    def test_baseline_concentrates_families_while_diversified_spreads(self):
        payload = self.base_payload()
        result = self.run_payload(payload)
        q0 = next(item for item in result["rankings"] if item["query_id"] == "query-0")
        base = next(s for s in q0["strategies"] if s["strategy_id"] == BASELINE_STRATEGY)
        diver = next(s for s in q0["strategies"] if s["strategy_id"] == DIVERSIFIED_STRATEGY)
        base_top10 = base["eligible_ranked"][:10]
        diver_top10 = diver["eligible_ranked"][:10]
        base_families = {row["family_id"] for row in base_top10}
        diver_families = {row["family_id"] for row in diver_top10}
        # 旧排序前十只覆盖 4 个家族且重复占位明显；去重策略覆盖全部 7 个可选家族。
        self.assertEqual(len(base_families), 4)
        self.assertEqual(len(diver_families), 7)
        base_dup = sum(1 for row in base_top10 if row["family_duplicate_of_rank"] is not None)
        diver_dup = sum(1 for row in diver_top10 if row["family_duplicate_of_rank"] is not None)
        self.assertGreater(base_dup, diver_dup)
        self.assertEqual(diver_dup, 3)  # 7 个家族轮询后才开始第二轮

    def test_unknown_never_treated_as_negative_and_recall_is_null(self):
        payload = self.base_payload()
        result = self.run_payload(payload)
        for strategy in result["budget_curves"]["strategies"]:
            for point in strategy["points"]:
                self.assertIsNone(point["development"]["micro"]["recall"])
                self.assertIsNone(point["heldout"]["micro"]["recall"])
                for entry in point["per_query"]:
                    self.assertIsNone(entry["recall"])
                    # 上界 = 确认相关 + 未知；下界只含确认相关。
                    self.assertEqual(
                        entry["confirmed_relevant_upper_bound"],
                        entry["confirmed_relevant"] + entry["unknown_in_review"],
                    )
        # 所有未裁决候选必须标 unknown，数据里没有负例。
        self.assertEqual(result["judgment_coverage"]["overall"]["negative_judgments"], 0)

    def test_holdout_families_excluded_from_development_split(self):
        payload = self.base_payload()
        result = self.run_payload(payload)
        heldout = set(payload["study"]["heldout_query_families"])
        leakage = next(
            check for check in result["validation"]["checks"] if check["check_id"] == "family_holdout_leakage"
        )
        self.assertTrue(leakage["passed"])
        self.assertEqual(leakage["detail"]["overlap"], [])
        for point in result["budget_curves"]["strategies"][0]["points"]:
            dev_families = {entry["family_id"] for entry in point["development"].get("per_query", [])}
            hold_families = {entry["family_id"] for entry in point["heldout"].get("per_query", [])}
            self.assertTrue(dev_families.isdisjoint(heldout))
            self.assertTrue(hold_families <= heldout)

    def test_selection_bias_is_quantified(self):
        payload = self.base_payload()
        result = self.run_payload(payload)
        bias = next(
            check for check in result["validation"]["checks"] if check["check_id"] == "selection_bias"
        )
        self.assertEqual(bias["detail"]["selector_counts"], {"old-similarity": 50})
        overall = result["judgment_coverage"]["overall"]
        self.assertLess(overall["judgment_coverage"], 0.2)
        self.assertEqual(overall["unknown_pairs"], overall["eligible_query_design_pairs"] - overall["judged_pairs"])
        # 配对比较必须指出改进策略独有候选没有标签。
        pair = next(
            item
            for item in result["budget_curves"]["paired_comparisons"]
            if item["budget"] == 10 and item["scope"] == "development"
        )
        for entry in pair["queries"]:
            self.assertEqual(entry["diversified_only"]["judged"], 0)
            self.assertEqual(entry["diversified_only"]["unknown"], entry["diversified_only"]["candidate_count"])

    def test_constrained_when_no_labels_at_all(self):
        payload = {
            "schema_version": 1,
            "as_of": "2026-10-30T00:00:00+00:00",
            "source_versions": {"operations": "t1"},
            "designs": [
                design("d1", "f1", "2026-08-01T00:00:00+00:00", [0.0, 0.0, 0]),
                design("d2", "f2", "2026-08-02T00:00:00+00:00", [1.0, 1.0, 1]),
            ],
            "queries": [query("q1", "f1", "2026-09-01T00:00:00+00:00", [0.1, 0.1, 0])],
            "judgments": [],
            "study": study([2, 5], ["f2"]),
            "baseline_case_id": None,
        }
        result = self.run_payload(payload)
        point = result["budget_curves"]["strategies"][0]["points"][0]
        entry = point["development"]["per_query"][0]
        self.assertIn("no_labeled_evidence_in_review", entry["constraint_reasons"])
        self.assertIn("all_reviewed_candidates_unknown", entry["constraint_reasons"])
        self.assertEqual(entry["confirmed_relevant"], 0)
        self.assertEqual(entry["confirmed_relevant_upper_bound"], entry["unknown_in_review"])
        self.assertIsNone(entry["judged_precision"])
        self.assertIsNone(entry["recall"])

    def test_constrained_when_all_candidates_single_family(self):
        payload = {
            "schema_version": 1,
            "as_of": "2026-10-30T00:00:00+00:00",
            "source_versions": {"operations": "t1"},
            "designs": [
                design("d1", "f1", "2026-08-01T00:00:00+00:00", [0.0, 0.0, 0]),
                design("d2", "f1", "2026-08-02T00:00:00+00:00", [0.2, 0.2, 1]),
            ],
            "queries": [query("q1", "f1", "2026-09-01T00:00:00+00:00", [0.1, 0.1, 0])],
            "judgments": [judgment("q1", "d1", True, "2026-09-10T00:00:00+00:00")],
            "study": study([5], ["f-other"]),
            "baseline_case_id": None,
        }
        result = self.run_payload(payload)
        entry = result["budget_curves"]["strategies"][0]["points"][0]["development"]["per_query"][0]
        self.assertIn("single_family_candidates", entry["constraint_reasons"])
        # 单家族时两套策略排序一致，去重无增益可宣称。
        base = [r["design_id"] for r in result["rankings"][0]["strategies"][0]["eligible_ranked"]]
        diver = [r["design_id"] for r in result["rankings"][0]["strategies"][1]["eligible_ranked"]]
        self.assertEqual(base, diver)

    def test_budget_exceeding_candidates_is_flagged(self):
        payload = {
            "schema_version": 1,
            "as_of": "2026-10-30T00:00:00+00:00",
            "source_versions": {"operations": "t1"},
            "designs": [design("d1", "f1", "2026-08-01T00:00:00+00:00", [0.0])],
            "queries": [query("q1", "f1", "2026-09-01T00:00:00+00:00", [0.1])],
            "judgments": [],
            "study": study([20], ["f2"]),
            "baseline_case_id": None,
        }
        result = self.run_payload(payload)
        entry = result["budget_curves"]["strategies"][0]["points"][0]["development"]["per_query"][0]
        self.assertIn("budget_exceeds_candidates", entry["constraint_reasons"])
        self.assertEqual(entry["reviewed"], 1)

    def test_late_judgments_are_visible_but_flagged(self):
        payload = self.base_payload()
        result = self.run_payload(payload)
        late = next(check for check in result["validation"]["checks"] if check["check_id"] == "late_labels")
        # query-8/9 的裁决在 2026-09-29 / 10-03，晚于 2026-09-25 cutoff。
        self.assertGreater(late["detail"]["late_judgment_count"], 0)
        self.assertTrue(all(item["available_at"] > payload["study"]["discovery_cutoff"] for item in late["detail"]["late_judgments"]))

    def test_development_query_cannot_see_late_judgment(self):
        cutoff = "2026-09-25T00:00:00+00:00"
        payload = {
            "schema_version": 1,
            "as_of": "2026-10-30T00:00:00+00:00",
            "source_versions": {"operations": "t1"},
            "designs": [
                design("d1", "f1", "2026-08-01T00:00:00+00:00", [0.0, 0.0, 0]),
                design("d2", "f2", "2026-08-02T00:00:00+00:00", [1.0, 1.0, 1]),
            ],
            "queries": [
                query("q1", "f1", "2026-09-01T00:00:00+00:00", [0.05, 0.05, 0]),
                query("q2", "f2", "2026-09-02T00:00:00+00:00", [0.95, 0.95, 1]),
            ],
            # q1（开发期）的裁决晚于 cutoff；q2 属留出家族，允许使用晚到标签。
            "judgments": [
                judgment("q1", "d1", True, "2026-09-30T00:00:00+00:00"),
                judgment("q2", "d2", True, "2026-09-30T00:00:00+00:00"),
            ],
            "study": study([2], ["f2"], cutoff=cutoff),
            "baseline_case_id": None,
        }
        result = self.run_payload(payload)
        q1 = next(q for q in result["rankings"] if q["query_id"] == "q1")
        q1_row = next(
            row for s in q1["strategies"] if s["strategy_id"] == BASELINE_STRATEGY for row in s["eligible_ranked"] if row["design_id"] == "d1"
        )
        self.assertEqual(q1_row["label_state"], "unknown")
        self.assertTrue(q1_row["label_masked_by_temporal_policy"])
        self.assertEqual(q1_row["raw_label_state"], "relevant")
        q2 = next(q for q in result["rankings"] if q["query_id"] == "q2")
        q2_row = next(
            row for s in q2["strategies"] if s["strategy_id"] == BASELINE_STRATEGY for row in s["eligible_ranked"] if row["design_id"] == "d2"
        )
        self.assertEqual(q2_row["label_state"], "relevant")
        self.assertFalse(q2_row["label_masked_by_temporal_policy"])
        self.assertEqual(result["judgment_coverage"]["overall"]["temporally_masked_judgments"], 1)

    def test_family_round_robin_keeps_within_family_distance_order(self):
        data = self.base_payload()
        designs = [DesignRecord.from_payload(item) for item in data["designs"]]
        q = QueryRecord.from_payload(data["queries"][0])
        ranked = rank_diversified(q, designs)
        # 每个家族内部仍按距离升序。
        seen: dict[str, float] = {}
        for item in ranked:
            previous = seen.get(item.design.family_id)
            if previous is not None:
                self.assertGreaterEqual(item.distance, previous)
            seen[item.design.family_id] = item.distance

    def test_duplicate_fingerprint_detection(self):
        d1 = DesignRecord.from_payload(design("d1", "f1", "2026-08-01T00:00:00+00:00", [0.1, 0.2, 1]))
        d2 = DesignRecord.from_payload(design("d2", "f2", "2026-08-02T00:00:00+00:00", [0.1, 0.2, 1]))
        self.assertEqual(d1.fingerprint_id, d2.fingerprint_id)
        q = QueryRecord.from_payload(query("q1", "f1", "2026-09-01T00:00:00+00:00", [0.1, 0.2, 1]))
        ranked = RANKERS[BASELINE_STRATEGY](q, [d1, d2])
        views = build_views(BASELINE_STRATEGY, q, ranked, {}, [d1, d2])
        self.assertIsNone(views[0].exact_duplicate_rank)
        self.assertEqual(views[1].exact_duplicate_rank, 1)

    def test_invalid_inputs_raise_study_input_error(self):
        good = self.base_payload()
        # 无时区时间。
        bad = deepcopy(good)
        bad["as_of"] = "2026-10-30T00:00:00"
        with self.assertRaises(StudyInputError):
            self.run_payload(bad)
        # 把未知写成 false 的语义错误由布尔类型通过；缺 design_id 必须报错。
        bad = deepcopy(good)
        del bad["designs"][0]["design_id"]
        with self.assertRaises(StudyInputError):
            self.run_payload(bad)
        # 空预算。
        bad = deepcopy(good)
        bad["study"]["review_budgets"] = []
        with self.assertRaises(StudyInputError):
            self.run_payload(bad)

    def test_judgment_referencing_unknown_entity_rejected(self):
        bad = self.base_payload()
        bad["judgments"][0]["design_id"] = "ghost"
        with self.assertRaises(StudyInputError):
            self.run_payload(bad)

    def test_no_infringement_wording_in_conclusions_and_limitations(self):
        result = self.run_payload(self.base_payload())
        self.assertIn("不认定", result["conclusions"]["no_infringement_assertion"])
        self.assertTrue(any("侵权" in item["text"] for item in result["limitations"]))

    def test_distance_and_similarity_monotonic(self):
        self.assertLess(euclidean_distance((0.0,), (0.1,)), euclidean_distance((0.0,), (0.9,)))
        near = CandidateView(
            "x", "f", "fp", 0.1, 0.9, 1, "unknown", None, None, None, None, None, True, None
        )
        # 数据类可构造，字段完整。
        self.assertEqual(near.label_state, "unknown")


if __name__ == "__main__":
    unittest.main()
