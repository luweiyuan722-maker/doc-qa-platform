# doc-qa-platform/ingest.py
# 文档索引工具 —— 把你的文档喂给向量库
#
# 用法：
#   python ingest.py --doc ./你的文档.pdf
#   python ingest.py --dir ./知识库文件夹/
#   python ingest.py --text "直接输入文本..."
#
# 支持的格式：.pdf, .txt, .md, .html

import os
import pickle
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

DATA_DIR = Path(__file__).parent / "data"
INDEX_FILE = DATA_DIR / "vector_index.pkl"

# ─── 中文友好的切分器 ──────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    length_function=len,
)


def load_file(file_path: str) -> list[Document]:
    """根据文件类型加载文档"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md", ".html", ".htm"):
        content = path.read_text(encoding="utf-8")
        return [Document(page_content=content, metadata={"source": path.name})]

    elif suffix == ".pdf":
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
            docs = PyMuPDFLoader(str(path)).load()
            for doc in docs:
                doc.metadata["source"] = path.name
            return docs
        except ImportError:
            print("  ⚠️  PDF 支持需要 pymupdf: pip install pymupdf langchain-community")
            return []

    else:
        print(f"  ⚠️  跳过不支持的文件格式: {suffix}")
        return []


def load_directory(dir_path: str) -> list[Document]:
    """递归加载目录中的所有支持文件"""
    all_docs = []
    for file_path in sorted(Path(dir_path).rglob("*")):
        if file_path.is_file():
            docs = load_file(str(file_path))
            if docs:
                print(f"  ✅ 加载: {file_path.name} ({len(docs)} 页/段)")
            all_docs.extend(docs)
    return all_docs


def main():
    parser = argparse.ArgumentParser(description="📚 文档索引工具 - 把知识喂给向量库")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doc",  type=str, help="单个文档路径")
    group.add_argument("--dir",  type=str, help="文档目录（递归加载）")
    group.add_argument("--text", type=str, help="直接输入文本")
    parser.add_argument("--chunk-size", type=int, default=500, help="切分大小（默认500）")
    parser.add_argument("--overlap",    type=int, default=50,  help="重叠大小（默认50）")

    args = parser.parse_args()

    # 检查 API Key
    if not os.getenv("ZHIPUAI_API_KEY"):
        print("❌ 请先设置 ZHIPUAI_API_KEY 环境变量")
        return

    # 自定义切分参数
    if args.chunk_size != 500 or args.overlap != 50:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
            length_function=len,
        )
    else:
        splitter = text_splitter

    # ─── 加载文档 ────────────────────────────────────
    print("📥 加载文档...")
    raw_docs = []

    if args.doc:
        raw_docs = load_file(args.doc)
    elif args.dir:
        raw_docs = load_directory(args.dir)
    elif args.text:
        raw_docs = [Document(page_content=args.text, metadata={"source": "手动输入"})]

    if not raw_docs:
        print("❌ 没有加载到任何文档")
        return

    # ─── 切分 ────────────────────────────────────────
    print(f"\n✂️  切分文档 (chunk_size={splitter._chunk_size}, overlap={splitter._chunk_overlap})...")
    chunks = splitter.split_documents(raw_docs)
    print(f"  📊 {len(raw_docs)} 原始文档 → {len(chunks)} 个 chunk")

    # ─── Embedding + 存储 ────────────────────────────
    print(f"\n🧠 向量化中 (智谱 embedding-2)...")
    # Embedding：智谱 AI（OpenAI 兼容接口）
    embeddings = OpenAIEmbeddings(
        model="embedding-2",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=os.getenv("ZHIPUAI_API_KEY"),
    )
    
    chunks_text = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(chunks_text)
    print(f"  ✅ 完成 {len(vectors)} 个向量")

    # ─── 保存到文件 ──────────────────────────────────
    print(f"\n💾 保存向量索引...")
    DATA_DIR.mkdir(exist_ok=True)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump({
            "docs": chunks,
            "embeddings": vectors,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
        }, f)

    file_size = INDEX_FILE.stat().st_size
    print(f"  ✅ 已保存到: {INDEX_FILE}")
    print(f"  📦 文件大小: {file_size / 1024:.1f} KB")

    print(f"\n🎉 完成！已索引 {len(chunks)} 个知识片段")
    print(f"\n🚀 现在可以启动问答服务: python server.py")


if __name__ == "__main__":
    main()
