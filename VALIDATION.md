# 交付验证记录

- 环境：Windows、Python 3.11.5，基础依赖版本见 requirements.lock.txt。
- `python -m unittest discover -s tests -v`：24 项通过。
- `python -m pip check`：No broken requirements found。
- CLI 示例建库：5 个文档成功入库。
- CLI 检索与问答演示：能定位配置文档并展示原文，明确标记 demo，不伪装成模型答案。
- 22 道示例问题已执行 vector / keyword / hybrid 检索评测；演示哈希向量的结果只验证管线与评测脚本。
- Streamlit AppTest：实际执行页面、点击建库、提交问题、生成来源面板。
- 实际 LangChain OpenAI-compatible Embedding / Chat 客户端：通过本地 HTTP 模拟服务验证请求与响应、Token 统计和回答引用。

未执行真实 DeepSeek 付费调用，未下载/运行本地 HuggingFace 语义模型，因此不声称已完成真实模型质量验收。
请使用自己的 Key 和实际 Embedding 模型完成 MANUAL_TESTS.md 的正式模式部分。
