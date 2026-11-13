# 业务演练数据口径

features 是本地数值指纹，无外部图像或模型依赖。judgments 只覆盖部分 query-design 对；不在表中即未知，绝非相关性 False。review_budgets 为每条查询的容量上限，family_id 表示衍生设计同族。selected_by 用于识别现有人工送审选择机制。

各 ID 均为本地合成标识，不包含真实人员或生产数据。时刻使用带时区 ISO 8601；as_of 是分析可知信息的截止时刻，不是业务发生时间上界。晚到事实允许保留在输入中，但只能在 available_at 到达之后采用。source_versions 标识输入版本；baseline_case_id 关联同库既有演示运行。空值表示未知，不应未经说明转成零。
