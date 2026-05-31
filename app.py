import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import sys
import subprocess
import streamlit as st
import pandas as pd

from src.retriever import load_model, load_faiss_index, load_metadata, retrieve
from src.rag_utils import generate_answer_with_openai, generate_answer_with_ollama
from src.guardrails import check_category_coverage



INDEX_PATH = "vector_store/faiss.index"
METADATA_PATH = "vector_store/metadata.pkl"
DATA_PATH = "data/institute_info_final.csv"


st.set_page_config(
    page_title="研究所入學小幫手",
    page_icon="🎓",
    layout="wide",
)


def check_files():
    missing = []
    if not os.path.exists(DATA_PATH):
        missing.append(DATA_PATH)
    return missing


def check_index_ready():
    return os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH)


def auto_build_index():
    """如果尚未建立 index，就自動執行 build_index.py。"""
    result = subprocess.run(
        [sys.executable, "build_index.py"],
        capture_output=True,
        text=True,
    )
    return result


@st.cache_resource
def cached_model():
    return load_model()


@st.cache_resource
def cached_index():
    return load_faiss_index(INDEX_PATH)


@st.cache_data
def cached_metadata():
    return load_metadata(METADATA_PATH)


def main():
    st.title("🎓 研究所入學 RAG 小助理")
    st.write(
        "這是中正語言所的入學小幫手！歡迎各位考生或是新生提出問題。🙋"
    )

    missing = check_files()
    if missing:
        st.error("缺少必要檔案：")
        for file in missing:
            st.code(file)
        st.stop()

    if not check_index_ready():
        st.warning("建立向量索引，系統正在自動建立。可能需要一點時間。")
        with st.spinner("正在建立 FAISS index..."):
            result = auto_build_index()

        if result.returncode != 0:
            st.error("自動建立 index 失敗。請把處理錯誤訊息")
            st.markdown("### 錯誤訊息")
            st.code(result.stderr or result.stdout)
            st.stop()
        else:
            st.success("向量索引建立完成！請重新整理頁面，或再次執行 `streamlit run app.py`。")
            st.code(result.stdout)
            st.stop()

    try:
        model = cached_model()
        index = cached_index()
        metadata = cached_metadata()
    except Exception as e:
        st.error("載入模型或索引失敗。請確認已安裝 requirements.txt 中的套件。")
        st.code(str(e))
        st.stop()

    categories = sorted(set(item["category"] for item in metadata))
    categories = ["全部"] + categories

    with st.sidebar:
        st.header("查詢設定")
        category_filter = st.selectbox("資料類別", categories)
        top_k = st.slider("檢索筆數", min_value=1, max_value=10, value=5)
        llm_mode = st.selectbox(
            "回答模式",
            ["只顯示檢索結果", "Ollama 本地 LLM", "OpenAI API"]
        )

#         ollama_model = st.selectbox(
#             "Ollama 模型",
#             ["llama3.2:3b", "qwen3:4b"]
# )

        st.markdown("---")
        st.caption("若要使用 GPT，請在 `.env` 中設定 OPENAI_API_KEY。")


    sample_questions = [
        "什麼是語言學？",
        "我對華語教學有興趣，需要實習嗎？",
        "畢業需要完成哪些事情？",
        "跨領域學生適合申請嗎？",
        "我想做語音分析，可以修哪些課？",
        "新生要怎麼找研究方向？",
        "有哪些組別？"
    ]

    st.subheader("請輸入你的問題")

    selected_sample = st.selectbox("常見問題", [""] + sample_questions)

    question = st.text_input(
        "問題",
        value=selected_sample,
        placeholder="例如：我對華語教學有興趣，需要實習嗎？",
    )

    if st.button("送出查詢", type="primary"):
        if not question.strip():
            st.warning("請先輸入問題。")
            st.stop()

        guardrail_result = check_category_coverage(question, metadata)
        if not guardrail_result["can_answer"]:
            st.warning(guardrail_result["reason"])
            # with st.expander("為什麼系統拒答？"):
            #     st.write("系統先判斷這個問題是否屬於目前資料庫可回答的範圍。")
            # st.json(guardrail_result)

            st.stop()

        results = retrieve(
            query=question,
            model=model,
            index=index,
            metadata=metadata,
            top_k=top_k,
            category_filter=category_filter,
        )

        if not results:
            st.info("目前找不到相關資料。你可以換個問法，或確認資料庫是否包含相關內容。")
            st.stop()

        if llm_mode == "Ollama 本地 LLM":
            st.markdown("## 🤖 小助理回答")
            with st.spinner("正在使用 Ollama 本地模型產生回答..."):
                answer = generate_answer_with_ollama(
                    question=question,
                    results=results,
                    model_name="llama3.2:3b"  # 這裡可以改成 ollama_model 變數，讓使用者選擇不同模型
                )
            st.write(answer)

        elif llm_mode == "OpenAI API":
            st.markdown("## 🤖 小助理回答")
            with st.spinner("正在使用 OpenAI 產生回答..."):
                answer = generate_answer_with_openai(question, results)
            st.write(answer)

        else:
            st.markdown("## 🔎 語意檢索結果")
            st.info("目前只顯示檢索結果，尚未啟用 LLM。")

        for i, item in enumerate(results, start=1):
            with st.container(border=True):
                st.markdown(f"### {i}. {item['title']}")
                st.write(f"**類別：** {item['category']}")

                search_type = item.get("search_type", "embedding")

                if search_type == "keyword":
                    st.write(f"**搜尋方式：** 關鍵字搜尋")
                    st.write(f"**關鍵字分數：** {item['score']:.1f}")
                else:
                    st.write(f"**搜尋方式：** 向量語意搜尋")
                    st.write(f"**相似度分數：** {item['score']:.3f}")

                st.write(item["content"])
                st.caption(f"來源：{item['source']}")

                if item["url"]:
                    st.markdown(f"[查看來源]({item['url']})")

    st.markdown("---")
    st.markdown("### 資料預覽")
    with st.expander("查看目前資料庫內容"):
        df = pd.DataFrame(metadata)
        st.dataframe(df[["id", "category", "title", "content", "source", "url"]], use_container_width=True)


if __name__ == "__main__":
    main()
