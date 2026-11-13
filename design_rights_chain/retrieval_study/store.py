"""检索策略研究的持久化与重放。

与既有案件库共用同一个 SQLite 文件，但使用独立表，不修改旧案件流的
``runs/events/idempotency/failures`` 表与人工裁决数据。
研究记录通过 ``baseline_case_id`` 关联同库既有演示案件，通过指纹编号
关联输入中的原稿/候选指纹。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from design_rights_chain.custody.case_store import stable_json, utc_now


class RetrievalStudyStore:
    def __init__(self, database: str | Path):
        self.database = str(database)
        self._initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS retrieval_studies(
                    study_id TEXT PRIMARY KEY,
                    input_fingerprint TEXT NOT NULL,
                    baseline_case_id TEXT,
                    baseline_case_present INTEGER NOT NULL,
                    as_of TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS retrieval_rankings(
                    study_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    design_id TEXT NOT NULL,
                    family_id TEXT NOT NULL,
                    fingerprint_id TEXT NOT NULL,
                    rank_position INTEGER NOT NULL,
                    eligible INTEGER NOT NULL,
                    label_state TEXT NOT NULL,
                    PRIMARY KEY(study_id, query_id, strategy_id, design_id)
                );
                CREATE INDEX IF NOT EXISTS retrieval_rankings_study ON retrieval_rankings(study_id);
                CREATE TABLE IF NOT EXISTS baseline_case_links(
                    study_id TEXT NOT NULL,
                    baseline_case_id TEXT NOT NULL,
                    case_present INTEGER NOT NULL,
                    linked_run_id TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(study_id, baseline_case_id)
                );
                """
            )

    def recall(self, study_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM retrieval_studies WHERE study_id=?",
                (study_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["result_json"])

    def baseline_case_status(self, baseline_case_id: str | None) -> dict[str, Any]:
        if not baseline_case_id:
            return {"baseline_case_id": None, "case_present": False, "linked_run_id": None}
        with self.connect() as connection:
            has_runs = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'"
            ).fetchone()
            if has_runs is None:
                return {
                    "baseline_case_id": baseline_case_id,
                    "case_present": False,
                    "linked_run_id": None,
                    "note": "案件流表尚未创建，无法关联原案件；先运行 casework.command 后可补链",
                }
            row = connection.execute(
                "SELECT run_id FROM runs WHERE case_id=? ORDER BY revision DESC LIMIT 1",
                (baseline_case_id,),
            ).fetchone()
        return {
            "baseline_case_id": baseline_case_id,
            "case_present": row is not None,
            "linked_run_id": row["run_id"] if row else None,
        }

    def save(self, study_id: str, input_fingerprint: str, result: Mapping[str, Any], rankings: Sequence[dict[str, Any]]) -> None:
        payload = stable_json(result)
        baseline = result.get("case_linkage", {})
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO retrieval_studies(study_id,input_fingerprint,baseline_case_id,baseline_case_present,as_of,created_at,result_json)"
                " VALUES(?,?,?,?,?,?,?) ON CONFLICT(study_id) DO UPDATE SET result_json=excluded.result_json, created_at=excluded.created_at",
                (
                    study_id,
                    input_fingerprint,
                    baseline.get("baseline_case_id"),
                    1 if baseline.get("case_present") else 0,
                    result.get("as_of"),
                    utc_now(),
                    payload,
                ),
            )
            connection.execute(
                "DELETE FROM retrieval_rankings WHERE study_id=?",
                (study_id,),
            )
            connection.executemany(
                "INSERT INTO retrieval_rankings(study_id,query_id,strategy_id,design_id,family_id,fingerprint_id,rank_position,eligible,label_state)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                [
                    (
                        study_id,
                        item["query_id"],
                        item["strategy_id"],
                        item["design_id"],
                        item["family_id"],
                        item["fingerprint_id"],
                        item["rank_position"],
                        1 if item["eligible"] else 0,
                        item["label_state"],
                    )
                    for item in rankings
                ],
            )
            connection.execute(
                "INSERT INTO baseline_case_links(study_id,baseline_case_id,case_present,linked_run_id,created_at)"
                " VALUES(?,?,?,?,?) ON CONFLICT(study_id, baseline_case_id) DO UPDATE SET case_present=excluded.case_present, linked_run_id=excluded.linked_run_id",
                (
                    study_id,
                    baseline.get("baseline_case_id") or "",
                    1 if baseline.get("case_present") else 0,
                    baseline.get("linked_run_id"),
                    utc_now(),
                ),
            )
