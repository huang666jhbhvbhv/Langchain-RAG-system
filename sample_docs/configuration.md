# AtlasDocs 配置

所有配置从应用目录 .env 加载，进程环境变量优先。不要把真实密钥提交到 Git。
LLM_API_KEY 是回答模型密钥；LLM_BASE_URL 是回答 API 地址；LLM_MODEL 是模型名称。
演示部署的回答模型接 DeepSeek。修改 .env 后重启应用。

Embedding 与回答模型独立配置。EMBEDDING_PROVIDER=local 使用本地模型；
EMBEDDING_PROVIDER=openai 使用 OpenAI-compatible 向量化接口，需要独立的
EMBEDDING_API_KEY、EMBEDDING_BASE_URL 和 EMBEDDING_MODEL。
聊天接口不等于 Embedding 接口，不能只换聊天 Key 就完成向量化适配。

CHUNK_SIZE 默认 800 字符，CHUNK_OVERLAP 默认 120 字符；切分按字符不是按 Token。
TOP_K 默认 5；CONTEXT_CHARS 默认 10000。修改分块参数或向量模型后要完整重建索引。
