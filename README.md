# Doc-QA 文档问答平台（RAG）

基于 LangChain + FastAPI 从零构建的 RAG 文档问答平台，打通「文档加载 → 切分 → 向量化 → 语义检索 → 生成」全链路，支持 SSE 流式输出。

## ✨ 核心特性

- **多格式文档加载**：支持 PDF / txt / md / html
- **中文友好切分**：RecursiveCharacterTextSplitter，按段落、换行、中文标点（。！？；）递归切分，chunk 大小可调
- **Embedding 向量化**：智谱 AI `embedding-2`
- **混合检索**：BM25 关键词 + 向量语义（余弦相似度），归一化后加权融合，提升召回准确率
- **防幻觉生成**：DeepSeek `deepseek-chat`，严格基于检索到的文档回答，无相关信息时明确拒绝编造
- **SSE 流式输出**：先返回检索来源，再逐 token 流式生成
- **多轮会话管理**：session_id 区分会话，历史上下文注入 + 滑动窗口（防无限增长）
- **前端页面 + Docker 容器化部署**

## 🛠 技术栈

Python · FastAPI · LangChain · DeepSeek · 智谱 Embedding · numpy · SSE · Docker

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量（参考 .env.example，填入你的 API Key）
cp .env.example .env

# 3. 索引文档（构建向量库）
python ingest.py --doc knowledge.md

# 4. 启动服务
python server.py

# 5. 打开浏览器
#    http://localhost:8000
```

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查（返回向量库状态 + 文档数） |
| POST | `/api/ask` | 普通问答（一次性返回答案 + 来源） |
| POST | `/api/ask/stream` | SSE 流式问答（先发来源，再逐 token 输出） |

## 📁 目录结构

```
doc-qa-platform/
├── ingest.py          # 文档索引工具（加载 → 切分 → 向量化 → 保存）
├── server.py          # FastAPI 服务（检索 + 生成 + SSE 流式）
├── knowledge.md       # 示例知识库文档
├── templates/
│   └── index.html     # 前端页面
├── data/              # 向量索引（运行时生成，不提交）
├── Dockerfile         # 容器化部署
├── requirements.txt   # 依赖
└── .env.example       # 环境变量模板
```

## 🏗 数据流

```
文档 ──加载──> 切分 ──embedding──> 向量库（pickle 持久化）
                                      │
用户问题 ──> BM25 关键词 + 向量语义（余弦相似度）──> 加权融合 Top-K ──> 拼 prompt ──> LLM ──> 流式返回
```

## 📄 License

MIT
