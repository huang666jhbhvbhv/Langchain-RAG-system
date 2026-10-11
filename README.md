# Project Docs RAG：基于 LangChain 的项目文档问答系统

一个可运行的本地项目文档 RAG（Retrieval-Augmented Generation，检索增强生成）系统。用户可以导入 Markdown、TXT 和文本型 PDF，系统完成文档切分、Embedding、混合检索，再由 DeepSeek 根据检索证据生成带来源引用的答案。

## 项目亮点

- **完整 RAG 链路**：使用 LangChain Document、RecursiveCharacterTextSplitter、Embeddings、ChatPromptTemplate 和 Runnable 组织导入、检索与生成流程。
- **混合检索**：实现向量余弦检索、BM25 关键词检索和 RRF 融合，兼顾语义相似度与精确关键词匹配。
- **可追溯回答**：记录来源文件、行号和 PDF 页码，校验模型引用编号；无足够证据时拒答。
- **增量知识库**：SQLite 持久化文件、片段和向量，使用 SHA-256 检测变化，只重新处理新增或修改的文档，并同步删除文件。
- **工程化交付**：提供 Streamlit 网页、命令行、离线演示、22 道评测题和自动化测试，支持本地 Embedding 或 OpenAI-compatible Embedding 服务。

## 技术栈

Python 3.11+、LangChain、DeepSeek API、HuggingFace Sentence Transformers、NumPy、SQLite、BM25、Streamlit、Python `unittest`。

## 系统流程

```text
文档目录 → Document → 文本分块 → Embedding → SQLite 索引
问题 → 向量检索 + BM25 → RRF 排名 → Prompt 注入证据 → DeepSeek → JSON 答案与引用
```

## 目录结构

```text
app.py                       Streamlit 网页入口
rag/                         配置、导入、向量、存储、检索、回答和评测模块
sample_docs/                 示例知识库文档
evaluation/                  评测问题集
tests/                       自动化测试
docs/rag_beginner_guide.tex  XeLaTeX 编写的原理与代码讲解
MANUAL_TESTS.md              手动验收步骤
```

`.env`、`.venv`、`.rag-data` 和评测报告属于本地运行数据，已通过 `.gitignore` 排除；仓库只提供 `.env.example` 配置模板。

独立于之前的编程 Agent。把 Markdown、TXT、文本 PDF 建成知识库，使用向量/BM25/混合检索找到证据，
再由 DeepSeek 回答，并展示来源文件、行号或 PDF 页码。含网页界面、命令行、增量同步和评测脚本。

学习来源是你提供的 `../参考代码/rag_from_scratch_1_to_4.ipynb`，以及
[LangChain rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)。
将 Notebook 的加载、切分、检索、生成流程重新组织为独立模块，使用当前 LangChain 包接口；没有导入或修改原 Notebook。

## 1. 第一次启动：无需密钥的演示

Windows / Python 3.11+。在 PowerShell 中执行（无需激活虚拟环境）：

```powershell
cd E:\code_learning\learn-cc\langchain-RAG\project-docs-rag
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

也可以依次双击 `setup.cmd` 和 `start.cmd`。已有 `.venv` 时直接 `start.cmd`。
浏览器打开 <http://localhost:8501>，默认只监听本机。

1. 保持左侧“离线演示”开启。
2. 保持默认示例文档目录，点击“同步 / 更新索引”。
3. 输入：`LLM_API_KEY 在哪里配置？`、`修改向量模型后怎么办？`
4. 展开来源，核对原文、行号和检索分数。

**离线模式使用词法哈希向量，只展示相关原文，不调用 LLM、不做语义生成，不能当成正式 RAG 质量测试。**
演示索引和正式索引使用不同目录，防止混用向量。

如果 pip 提示找不到新版本，可明确换用可访问的源：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt --index-url https://pypi.org/simple
```

`requirements.lock.txt` 固定本次测试的完整基础依赖；`requirements.txt` 是允许升级的直接依赖范围。
退出网页服务：在启动终端按 Ctrl+C。页面聊天在浏览器会话中保存，可导出 JSON；不是长期聊天数据库。

## 2. 正式模式：DeepSeek + 独立 Embedding

首次创建配置（不要覆盖你已有的 .env）：

```powershell
Copy-Item .env.example .env
notepad .env
```

填写聊天模型：

```ini
LLM_API_KEY=你的DeepSeek密钥
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
```

模型名称以你的服务商实际支持为准，修改 `LLM_MODEL` 即可。Base URL 填根地址或 `/v1` 根路径，
不重复添加 `/chat/completions`。兼容平台差异仍需验证。
不会读取或复制另一个 deepseek-code 项目的 .env。应用目录 .env 的优先级低于现有环境变量。

### 选择 A：本地向量模型（不另付 Embedding API 费）

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt -c requirements.lock.txt
```

配置默认就是：

```ini
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

通过 LangChain HuggingFaceEmbeddings 在 CPU 运行多语言模型。首次需要下载模型和 PyTorch 等较大依赖；
需要可访问模型仓库的网络。也可将 `EMBEDDING_MODEL` 改为已完整下载的 Sentence Transformers 模型目录。
首次加载较慢，切勿把下载等待误认为程序死循环。更换模型/模型文件后应完整重建索引。
该模型适合入门，长片段可能被模型截断；需根据实际文档调小 CHUNK_SIZE 或换长上下文 Embedding 模型并评测。

### 选择 B：远程 OpenAI-compatible Embedding

不用安装本地模型依赖，配置独立服务：

```ini
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://你的向量服务地址/v1
EMBEDDING_API_KEY=该向量服务的密钥
EMBEDDING_MODEL=该服务实际提供的向量模型名称
```

**聊天 API 和 Embedding API 是两种能力。不能只把 DeepSeek 聊天地址填进 Embedding 配置就认为能用。**
远程向量化会发送文档片段和问题给你配置的服务；聊天请求会发送选中的文档片段给回答模型。

配置完成后重新启动网页，关闭“离线演示”，点击同步索引，再提问。
回答中的 `[1]` 等编号对应下方来源。模型拒答、引用格式错误和无检索证据有不同状态提示。

## 3. 换成你自己的项目文档

建议新建一个只存公开/可发送资料的文件夹，放 README、安装说明、配置手册和 FAQ。
左侧填写该文件夹的绝对路径，或者上传文件后点击“保存上传文件到文档目录”，再同步索引。
上传不会自动覆盖同名文件，需明确勾选覆盖。

每个知识库名绑定一个文档根目录；换项目请更换知识库名。
删除源文件后再次同步，会移除其旧索引；修改源文件只重新向量化变化的文件。
更改 Embedding 或分块配置时勾选“完整重建”，重建失败仍保留旧索引，但查询时需使用匹配旧索引的配置。
解析失败时整个同步中止，不会把读取失败的文档误当成被删除。

支持 UTF-8/UTF-8-BOM 的 MD/TXT，PDF 需要文本层；扫描 PDF 不做 OCR。
默认跳过隐藏文件/目录、.git、虚拟环境和符号链接。单文件 20 MB、PDF 500 页、扫描 2000 个文件，
单次更新最多 20000 个片段。PDF 复杂排版的提取顺序仍可能影响回答；页码只表示来源页。

## 4. 命令行

全局参数 `--demo`、`--data-dir` 放在子命令前。省略 `--demo` 为正式配置。

```powershell
# 环境检查：不联网、不打印 Key
.\.venv\Scripts\python.exe -m rag doctor
# 离线示例建库、检索、展示原文
.\.venv\Scripts\python.exe -m rag --demo index sample_docs
.\.venv\Scripts\python.exe -m rag --demo search "LLM_API_KEY" --mode hybrid
.\.venv\Scripts\python.exe -m rag --demo ask "如何启动网页？"
# 正式索引和问答
.\.venv\Scripts\python.exe -m rag index sample_docs
.\.venv\Scripts\python.exe -m rag ask "更换模型后需要做什么？"
# 配置变化后完整重建
.\.venv\Scripts\python.exe -m rag index sample_docs --rebuild
# 不同项目使用独立索引
.\.venv\Scripts\python.exe -m rag --data-dir .rag-data\my-project index "E:\你的文档目录"
.\.venv\Scripts\python.exe -m rag --data-dir .rag-data\my-project ask "项目怎样启动？"
```

网页默认使用 `.rag-data/default-demo` 或 `default-live`，CLI 默认使用 `.rag-data/demo` 或 `default`。
若希望网页和 CLI 共用索引，CLI 用 `--data-dir .rag-data/default-live` 等指定对应路径。

## 5. 架构与关键代码

```text
导入 → LangChain Document → RecursiveCharacterTextSplitter
                               ↓
                       Embeddings 批量向量化
                               ↓
                  SQLite 原文/向量/文件哈希事务提交

问题 → 可选多轮改写 → 向量余弦 + BM25 → RRF 排名融合
                                          ↓
                  上下文长度限制 → ChatPromptTemplate
                                          ↓
                     ChatOpenAI（DeepSeek 兼容 API）
                                          ↓
                     JSON 解析 → 引用校验 → 答案/来源
```

- `rag/config.py`：独立配置、分块参数校验、索引配置指纹。
- `rag/ingest.py`：MD/TXT/PDF 解析与分块，记录准确字符位置、行号、页码。
- `rag/embedding.py`：LangChain Embeddings 接口，本地模型/远程 API/显式演示模式。
- `rag/store.py`：SQLite 持久化、SHA-256 增量检测、失败回滚、文件锁与读取快照。
- `rag/retrieve.py`：NumPy 精确余弦搜索、中文字符/二元组及英文标识符 BM25、RRF。
- `rag/answer.py`：LangChain Runnable 组合、Prompt、模型调用、JSON 与引用检查、用量统计。
- `app.py`：Streamlit 导入、索引管理、问答、原文展开、对话导出。
- `rag/evaluate.py`：三种检索方式的评测与报告。

这里真正使用了 LangChain 的 Document、Splitter、Embeddings、ChatOpenAI、Prompt、Runnable 和 Parser。
原 Notebook 使用 Chroma，本项目存储层改为 SQLite + NumPy 精确检索，便于在 Windows 进行可恢复的原子更新。
它不是 Chroma/FAISS 的 ANN 索引，查询会加载知识库片段并扫描全部向量，适合本地中小型库，不适合百万级文档。

## 6. 评测：不要只看模型说“回答正确”

附 22 道示例题，其中 18 道有文档依据、4 道无答案题。示例文档是虚构的 AtlasDocs 项目。
固定同一题集，对比 vector、keyword、hybrid，默认评测 top_k=3：

```powershell
.\.venv\Scripts\python.exe -m rag --demo index sample_docs
.\.venv\Scripts\python.exe -m rag --demo evaluate --output reports/demo-evaluation.json
# 正式语义模型：先建正式索引，再评测
.\.venv\Scripts\python.exe -m rag index sample_docs
.\.venv\Scripts\python.exe -m rag evaluate --output reports/retrieval-evaluation.json
# 额外生成每题答案：会调用真实聊天 API，三种模式共最多 66 次回答请求
.\.venv\Scripts\python.exe -m rag evaluate --answers --output reports/answer-evaluation.json
```

报告包括 source-level Hit@K、来源召回率、MRR、无答案题的空检索比例、平均耗时；
`--answers` 增加模型答案、引用编号、Token 用量和参考词覆盖率。
这些指标不等同于答案正确率。无答案题可能检索到相关但不足的片段，应继续评估模型是否拒答。
引用编号合法只说明能定位到检索原文，不能证明每个回答结论被原文支持。
5 个短示例文档不足以形成有说服力的能力结论，请扩充真实项目问题、片段级证据和人工评分。
不要把演示向量或这份小样本的成绩写成生产效果提升。

## 7. 自动测试与手动验收

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

自动测试不需要真实 Key：覆盖持久化、增删改同步、不重复 Embedding、并发更新、失败回滚、
PDF 页码、准确行号、无依据拒答、非法引用、真实 LangChain HTTP 客户端对接本地模拟服务、网页建库和问答。
交付前在 Windows / Python 3.11.5 下通过 24 项测试，pip check 未发现依赖冲突。
真实 DeepSeek 联网问答和本地语义模型首次下载尚未在本次验证中执行；需按第 2 节配置后手测。
独立手测步骤见 [MANUAL_TESTS.md](MANUAL_TESTS.md)。

## 8. 边界与排错

- 401/404：核对对应服务的 Key、模型名、Base URL，别把聊天服务与向量服务混用。
- 模型下载失败：检查网络，或使用本地完整模型目录/远程 Embedding；系统不会静默退回演示向量。
- 引用无效：本次模型答案不会作为正常答案展示；可查看原文或重新提问。
- 文档更新未生效：需点击同步索引；正在进行的问答引用的是检索时刻的索引快照。
- 召回差：先对比关键词检索，检查原文是否入库，再调分块大小、Top-K 和阈值。
- 环境冲突：使用本项目独立 `.venv`，不要与旧教程全局依赖混装。
- 多轮问答：默认关闭，开启后用最近 4 轮上下文改写问题，额外调用一次模型；历史回答不当作事实证据。

目前没有 OCR、Word/Excel 解析、联网搜索、reranker、多租户鉴权、知识库加密和全局费用预算。
只作为本机单用户应用，不要直接暴露到公网。默认不开启 LangSmith tracing；若你设置了相关全局环境变量，
LangChain 可能按环境配置上传追踪信息，请自行检查。
生成模型能否遵循证据、是否正确拒答，仍需用真实模型评测；系统不能保证完全消除幻觉。
usage 统计聊天模型成功响应中的 Token，不包含 Embedding 费用、失败请求或服务端可能已计费的重试。
