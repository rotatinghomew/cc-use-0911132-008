"""检索策略研究包：在有限人工复核预算下安排复核队列。

公开入口：``run_study``。本包只读取本地合成输入与本地 SQLite，
不下载模型、不连接生产系统，也不产出侵权认定。
"""
from __future__ import annotations

from design_rights_chain.retrieval_study.engine import run_study
from design_rights_chain.retrieval_study.features import StudyInputError

__all__ = ["run_study", "StudyInputError"]
