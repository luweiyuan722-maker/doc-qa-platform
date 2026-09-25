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
    
    def load(self, path: Path):
        """从 pickle 文件加载"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.docs = data["docs"]
        self.vectors = np.array(data["embeddings"])
    
    @property
    def count(self) -> int:
        return len(self.docs)
    
    def search(self, query: str, k: int = 5) -> list[Document]:
        """余弦相似度搜索"""
        query_vec = np.array(embeddings_model.embed_query(query))
        
        # 归一化
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        vecs_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)
        
        # 余弦相似度
        similarities = np.dot(vecs_norm, query_norm)
        
        # Top-K
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            doc = self.docs[idx]
            doc.metadata["_score"] = float(similarities[idx])
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

## ❓ 用户问题
{question}

## ✅ 回答：""")

def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", f"文档片段{i+1}")
        parts.append(f"--- [来源{i+1}: {source}] ---\n{doc.page_content}")
    return "\n\n".join(parts)

# ─── API 模型 ────────────────────────────────────────
class Question(BaseModel):
    text: str
    k: int = 5

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
    prompt = RAG_PROMPT.format(context=context, question=question.text)
    result = llm.invoke(prompt)

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

    prompt = RAG_PROMPT.format(context=context, question=question.text)

    async def event_stream():
        yield f"data: {sources_data}\n\n"

        async for chunk in llm.astream(prompt):
            if chunk.content:
                data = json.dumps({"type": "token", "content": chunk.content}, ensure_ascii=False)
                yield f"data: {data}\n\n"

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
