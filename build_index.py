import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import pickle
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/institute_info_final.csv"
INDEX_PATH = "vector_store/faiss.index"
METADATA_PATH = "vector_store/metadata.pkl"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_data(data_path: str) -> pd.DataFrame:
    """讀取研究所資訊資料。"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到資料檔：{data_path}")

    df = pd.read_csv(data_path)
    required_cols = ["id", "category", "title", "content", "source", "url"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要欄位：{col}")

    df = df.fillna("")

    # 把類別、標題、內容都放進 embedding 文字，檢索效果通常會比較穩。
    df["text_for_embedding"] = (
        "類別：" + df["category"].astype(str) + "\n"
        + "標題：" + df["title"].astype(str) + "\n"
        + "內容：" + df["content"].astype(str)
    )

    return df


def build_index(df: pd.DataFrame):
    """建立 FAISS 向量索引。"""
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        df["text_for_embedding"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # 因為 embedding 已 normalize，所以 Inner Product 約等於 cosine similarity。
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def main():
    os.makedirs("vector_store", exist_ok=True)

    df = load_data(DATA_PATH)
    index = build_index(df)

    faiss.write_index(index, INDEX_PATH)

    metadata = df[
        ["id", "category", "title", "content", "source", "url", "text_for_embedding"]
    ].to_dict(orient="records")

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ 向量索引建立完成")
    print(f"資料筆數：{len(metadata)}")
    print(f"Index 路徑：{INDEX_PATH}")
    print(f"Metadata 路徑：{METADATA_PATH}")


if __name__ == "__main__":
    main()
