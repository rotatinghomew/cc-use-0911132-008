from __future__ import annotations

import argparse
import json
import sys

from design_rights_chain.retrieval_study import StudyInputError, run_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="design_rights_chain.retrieval_study",
        description=(
            "检索策略研究：在有限人工复核预算下比较旧相似度基线与家族去重改进策略，"
            "输出有效发现、重复占位、覆盖与不确定性；不做侵权认定。"
        ),
    )
    parser.add_argument("--input", required=True, help="分析输入 JSON（作品、查询、候选特征、送审与裁决）")
    parser.add_argument("--database", required=True, help="与既有案件流共用的 SQLite 数据库路径")
    parser.add_argument("--output", required=True, help="研究结果输出 JSON 路径")
    args = parser.parse_args(argv)
    try:
        result = run_study(args.input, args.database, args.output)
    except StudyInputError as exc:
        print(f"输入校验失败: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"study_id": result["study_id"], "output": args.output, "replayed": result["replayed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
