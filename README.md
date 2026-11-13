# 设计相似检索的复核容量与证据外推

项目使用 Python 3.11 标准库与 SQLite，现有逐案处理入口为 `design_rights_chain.casework.command`。保留原有处理器、持久化与测试；分析业务使用本地合成资料，不连接生产系统。无需 pip 安装、下载模型或启动额外服务。

## 现有程序

在仓库根目录运行 `bash scripts/demo.sh`，依次完成现有产品回归、分析输入核对、语法构建和真实业务入口调用。所有运行文件位于临时目录，结束后清理；运行失败立即退出，不使用预存输出代替执行。

单独测试：`python3 -m unittest discover -s tests/rights_cases -p 'test_*.py' -v`。

输入检查：`python3 -m unittest discover -s tests -p test_analysis_inputs.py -v`。

构建检查：`python3 -m compileall -q design_rights_chain tests`。

业务调用：`python3 -m design_rights_chain.casework.command fixtures/demo.json --database /tmp/design_rights_chain-demo.sqlite3`。

## 分析资料与联调

`fixtures/analysis/input.json` 保存业务演练数据，字段解释在 `fixtures/analysis/fields.md`。以下已核对的数据形态用于分析回归：partial_judgments、design_family、late_candidates。现有输入测试检查这些形态是否真实存在，不代替分析方法或数值正确性的验证。

分析入口接通后执行 `bash scripts/feature_demo.sh`：先重跑现有产品，再从干净数据库计算分析、同输入重放、核对输出结构并打印完整 JSON。尚未接通时该命令会报模块不存在，不会用静态样例冒充成功。分析 ID 为 `study_id`，其余输出包括 `rankings, budget_curves, judgment_coverage, validation, limitations`。方法正确性还需覆盖业务说明中的反例，结构核验不能代替统计验证。

## 检索策略研究（复核队列安排）

入口：`python3 -m design_rights_chain.retrieval_study --input fixtures/analysis/input.json --database <case_db> --output <retrieval_json>`。研究与旧案件流共用 SQLite，但只读写独立的 `retrieval_studies / retrieval_rankings / baseline_case_links` 表，不修改 `runs/events/idempotency/failures` 与人工裁决；`case_linkage` 通过 `baseline_case_id` 与指纹编号连回原案件和原稿记录。

* 基线 `old_similarity`：按本地数值指纹欧氏距离升序（相似度降序）排序，不去重，复刻"前十条全是同一原稿裁切版本"的旧行为。
* 改进 `family_diversified`：先按 `family_id` 聚合、以家族最近邻距离定家族顺序，再逐家族轮询，每轮每家族只放行一个候选，有限名额先覆盖更多原稿家族。
* 指标：有限预算下的已确认相关数（只计已审相关）、确认相关乐观上界（把未知全计为相关）、家族覆盖、精确/家族重复占位、未审占比；`recall` 恒为 `null`，因为真实相关总量不可识别。
* 口径红线：未审候选是 `unknown` 不是负例；查询时刻未入库的候选不可检出；晚于 `discovery_cutoff` 到达的裁决在开发期查询中屏蔽为未知（留出家族评估可用）；同家族改稿不跨开发/留出切分；输出不构成侵权认定。
* 受限结论：无相关性证据、候选全属同族、预算超过候选数等情形在 `validation.constrained_summary` 与每条 `per_query.constraint_reasons` 中显式标注，只给区间不给点估计。
* 选择偏差：标签只覆盖旧策略送审的候选（MNAR），`judgment_coverage` 量化覆盖比例，`budget_curves.paired_comparisons` 只在共享候选与标签无关指标上下结论，改进策略独有候选标注为待主动送审。

## 离线环境

使用 Python 3.11.2 的 Linux 环境即可复现。容器环境预置解释器后，以只读方式挂载仓库，将文件复制到容器临时目录再运行脚本；运行时关闭网络。基础测试和小规模本地数据适合在 90 秒内展示，正式展示命令为 `bash scripts/feature_demo.sh`。不要在展示脚本里安装依赖或等待外部服务。
