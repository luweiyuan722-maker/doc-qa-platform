# Doc-QA 平台 —— 完整 RAG + SSE 流式输出
#
# 模型配置：
#   Chat:     DeepSeek (deepseek-chat)
#   Embedding: 智谱 AI (embedding-2)
#
# 启动：
#   1. pip install -r requirements.txt
#   2. 创建 .env 文件（参考 .env.example）
#   3. python ingest.py --doc knowledge.md
#   4. python server.py
#   5. 打开 http://localhost:8000

import os, json, pickle, numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import jieba
from rank_bm25 import BM25Okapi

# ─── 配置 ───────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
INDEX_FILE = DATA_DIR / "vector_index.pkl"

# ─── 模型初始化 ──────────────────────────────────────────
# Chat：DeepSeek（OpenAI 兼容接口）
llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL", "deepseek-chat"),
    base_url=os.getenv("CHAT_BASE_URL", "https://api.deepseek.com/v1"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3,
    streaming=True,
)

# Embedding：智谱 AI（OpenAI 兼容接口）
embeddings_model = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "embedding-2"),
    base_url=os.getenv("EMBEDDING_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
    api_key=os.getenv("ZHIPUAI_API_KEY"),
)

app = FastAPI(title="📚 智能文档问答平台", version="1.0.0")

# ─── 简易向量存储（numpy + pickle） ──────────────────
class SimpleVectorStore:
    """轻量级向量存储：numpy 做相似度搜索，pickle 做持久化"""
    
    def __init__(self):
        self.docs: list[Document] = []
        self.vectors: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
    
    def load(self, path: Path):
        """从 pickle 文件加载"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.docs = data["docs"]
        self.vectors = np.array(data["embeddings"])
        # 构建 BM25 索引（jieba 中文分词）
        self.bm25 = BM25Okapi([list(jieba.cut(doc.page_content)) for doc in self.docs])
    
    @property
    def count(self) -> int:
        return len(self.docs)
    
    def search(self, query: str, k: int = 5) -> list[Document]:
        """混合检索：BM25 关键词 + 向量语义，归一化后加权融合"""
        # 1. 向量检索（余弦相似度）
        query_vec = np.array(embeddings_model.embed_query(query))
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        vecs_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)
        vec_scores = np.dot(vecs_norm, query_norm)

        # 2. BM25 关键词检索
        query_tokens = list(jieba.cut(query))
        bm25_scores = np.array(self.bm25.get_scores(query_tokens), dtype=float)

        # 3. 归一化 + 加权融合（0.5 向量 + 0.5 BM25）
        def _normalize(x: np.ndarray) -> np.ndarray:
            x_min, x_max = x.min(), x.max()
            if x_max - x_min < 1e-10:
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min)

        final_scores = 0.5 * _normalize(vec_scores) + 0.5 * _normalize(bm25_scores)

        # Top-K
        top_indices = np.argsort(final_scores)[::-1][:k]

        results = []
        for idx in top_indices:
            doc = self.docs[idx]
            doc.metadata["_score"] = float(final_scores[idx])
            doc.metadata["_vec_score"] = round(float(vec_scores[idx]), 4)
            doc.metadata["_bm25_score"] = round(float(bm25_scores[idx]), 4)
            results.append(doc)

        return results

_vectorstore = SimpleVectorStore()

def get_vectorstore() -> SimpleVectorStore:
    if _vectorstore.count == 0:
        if not INDEX_FILE.exists():
            raise HTTPException(
                status_code=503,
                detail="向量库尚未初始化，请先运行: python ingest.py --doc <你的文档>"
            )
        _vectorstore.load(INDEX_FILE)
    return _vectorstore

# ─── RAG 提示词 ──────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""你是一个专业的知识助手。请**仅根据以下提供的文档内容**回答问题。

如果文档中没有相关信息，请诚实地说"根据现有文档，我无法回答这个问题"，不要编造任何信息。

回答要求：
- 优先引用文档原文
- 如果涉及多个来源，请综合回答
- 标注信息来源（如"根据文档第X部分..."）

## 📄 相关文档内容
{context}

## 💬 对话历史
{history}

## ❓ 用户问题
{question}

## ✅ 回答：""")

def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", f"文档片段{i+1}")
        parts.append(f"--- [来源{i+1}: {source}] ---\n{doc.page_content}")
    return "\n\n".join(parts)

# ─── 多轮会话管理（内存版 + 滑动窗口）──────────────
_session_history: dict[str, list[dict]] = {}

def get_history(session_id: str) -> list[dict]:
    """取会话历史"""
    return _session_history.get(session_id, [])

def format_history(session_id: str, max_turns: int = 6) -> str:
    """把最近 N 轮对话格式化成文本（滑动窗口：只取最近 max_turns*2 条）"""
    history = get_history(session_id)
    if not history:
        return "（无历史对话）"
    recent = history[-(max_turns * 2):]
    return "\n".join(
        f"{'用户' if h['role'] == 'user' else '助手'}: {h['content']}" for h in recent
    )

def add_history(session_id: str, role: str, content: str, max_len: int = 40):
    """追加一条历史，超过 max_len 就滑动窗口丢弃最旧的（防无限增长）"""
    if session_id not in _session_history:
        _session_history[session_id] = []
    _session_history[session_id].append({"role": role, "content": content})
    if len(_session_history[session_id]) > max_len:
        _session_history[session_id] = _session_history[session_id][-max_len:]

# ─── API 模型 ────────────────────────────────────────
class Question(BaseModel):
    text: str
    k: int = 5
    session_id: str = "default"

# ─── 接口 1: 状态检查 ─────────────────────────────────
@app.get("/api/health")
async def health():
    try:
        vs = get_vectorstore()
        return {"status": "ready", "document_count": vs.count}
    except HTTPException:
        return {"status": "no_index", "document_count": 0}

# ─── 接口 2: 普通问答 ─────────────────────────────────
@app.post("/api/ask")
async def ask(question: Question):
    vs = get_vectorstore()
    docs = vs.search(question.text, k=question.k)

    if not docs:
        return {"answer": "未找到相关文档内容。", "sources": []}

    context = format_docs(docs)
    history_text = format_history(question.session_id)
    prompt = RAG_PROMPT.format(context=context, history=history_text, question=question.text)
    result = llm.invoke(prompt)

    # 更新会话历史（滑动窗口，最多 40 条）
    add_history(question.session_id, "user", question.text)
    add_history(question.session_id, "assistant", result.content)

    return {
        "answer": result.content,
        "sources": [
            {"index": i+1, "source": d.metadata.get("source", "未知"), "preview": d.page_content[:100]}
            for i, d in enumerate(docs)
        ]
    }

# ─── 接口 3: SSE 流式问答 ⭐────────────────────────────
@app.post("/api/ask/stream")
async def ask_stream(question: Question):
    vs = get_vectorstore()

    # 1. 检索
    docs = vs.search(question.text, k=question.k)
    context = format_docs(docs)

    # 2. 先发送来源
    sources_data = json.dumps({
        "type": "sources",
        "sources": [
            {"index": i+1, "source": d.metadata.get("source", "未知"), "preview": d.page_content[:80]}
            for i, d in enumerate(docs)
        ]
    }, ensure_ascii=False)

    history_text = format_history(question.session_id)
    prompt = RAG_PROMPT.format(context=context, history=history_text, question=question.text)

    async def event_stream():
        yield f"data: {sources_data}\n\n"

        full_answer = []  # 收集完整回答，流式结束后存历史
        async for chunk in llm.astream(prompt):
            if chunk.content:
                full_answer.append(chunk.content)
                data = json.dumps({"type": "token", "content": chunk.content}, ensure_ascii=False)
                yield f"data: {data}\n\n"

        # 流式结束后更新会话历史
        add_history(question.session_id, "user", question.text)
        add_history(question.session_id, "assistant", "".join(full_answer))

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# ─── 前端页面 ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
