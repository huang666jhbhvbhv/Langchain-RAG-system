# AtlasDocs 示例项目

这是用于学习 RAG 的虚构项目文档，以下参数只适用于 AtlasDocs 示例。

AtlasDocs 是团队项目文档问答工具。最低 Python 版本为 3.11。
安装依赖：python -m pip install -r requirements.txt。
启动网页：python -m streamlit run app.py。默认端口为 8501，默认只监听 127.0.0.1。
在启动终端按 Ctrl+C 退出，不会删除已建立的索引。

支持 Markdown、TXT 和有文本层的 PDF。每个文档大小不超过 20 MB。
扫描版 PDF 需要先 OCR，当前不包含 OCR 服务。暂不直接解析 Word 和 Excel。

用户应先导入文档并同步索引，再提问。修改或删除原文后必须再次同步索引。
