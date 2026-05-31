import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import pickle
import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_model():
    return SentenceTransformer(MODEL_NAME)


def load_faiss_index(index_path: str = "vector_store/faiss.index"):
    return faiss.read_index(index_path)


def load_metadata(metadata_path: str = "vector_store/metadata.pkl"):
    with open(metadata_path, "rb") as f:
        return pickle.load(f)



def normalize_text(text: str) -> str:
    return str(text).lower().strip()


def keyword_search(query: str, metadata, top_k: int = 5, category_filter: str = "全部"):
    """
    短 query 優先使用關鍵字搜尋。
    會搜尋 title、content、category、source。
    """
    query_norm = normalize_text(query)

    results = []

    for item in metadata:
        if category_filter != "全部" and item["category"] != category_filter:
            continue

        title = normalize_text(item.get("title", ""))
        content = normalize_text(item.get("content", ""))
        category = normalize_text(item.get("category", ""))
        source = normalize_text(item.get("source", ""))

        score = 0

        # title 命中最重要
        if query_norm in title:
            score += 3

        # content 命中次重要
        if query_norm in content:
            score += 2

        # category / source 命中也給一點分
        if query_norm in category:
            score += 1

        if query_norm in source:
            score += 1

        if score > 0:
            results.append({
                "score": float(score),
                "search_type": "keyword",
                "id": item["id"],
                "category": item["category"],
                "title": item["title"],
                "content": item["content"],
                "source": item["source"],
                "url": item["url"],
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]




def retrieve(
    query: str,
    model,
    index,
    metadata,
    top_k: int = 5,
    category_filter: str = "全部",
    short_query_keyword_first: bool = True,
    short_query_max_len: int = 3,
):
    """
    檢索流程：
    1. 如果 query 長度 <= short_query_max_len，先使用 keyword search。
    2. 如果 keyword search 有結果，直接回傳 keyword 結果。
    3. 如果 keyword search 沒結果，再使用 embedding / FAISS search。
    """

    clean_query = query.strip()

    # 短 query 先使用關鍵字搜尋
    if short_query_keyword_first and len(clean_query) <= short_query_max_len:
        keyword_results = keyword_search(
            query=clean_query,
            metadata=metadata,
            top_k=top_k,
            category_filter=category_filter,
        )

        if keyword_results:
            return keyword_results

    # 如果不是短 query，或 keyword search 沒找到，再走原本 FAISS
    query_embedding = model.encode(
        [clean_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k * 4)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = metadata[idx]

        if category_filter != "全部" and item["category"] != category_filter:
            continue

        results.append({
            "score": float(score),
            "search_type": "embedding",
            "id": item["id"],
            "category": item["category"],
            "title": item["title"],
            "content": item["content"],
            "source": item["source"],
            "url": item["url"],
        })

        if len(results) >= top_k:
            break

    return results
