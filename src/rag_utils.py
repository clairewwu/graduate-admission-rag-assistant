import os
from typing import List, Dict


def build_context(results: List[Dict]) -> str:
    """把檢索結果組成可以丟給 LLM 的 context。"""
    context_blocks = []

    for i, item in enumerate(results, start=1):
        block = f"""
[資料 {i}]
標題：{item['title']}
類別：{item['category']}
內容：{item['content']}
來源：{item['source']}
網址：{item['url']}
"""
        context_blocks.append(block.strip())

    return "\n\n---\n\n".join(context_blocks)


def build_prompt(question: str, context: str) -> str:
    """RAG prompt。"""
    return f"""
你是「中正語言所專門的研究所入學小助理」，任務是協助準研究生與新生理解研究所資訊。

請根據「可參考資料」回答使用者問題。

回答規則：
1. 只能根據可參考資料回答，不要自行編造老師、課程、學分、規則或招生資訊。
2. 其他學校也不要提，若使用者特別問到，請說只回答在中正語言所的問題。
3. 如果資料不足，請明確說：「目前資料不足，建議查詢系所官網或詢問系辦。」
4. 回答要清楚、友善，適合準備報考或剛入學的學生閱讀。
5. 若問題涉及年度、名額、學分、考試方式等可能變動的資訊，請提醒使用者以最新公告為準。
6. 若是問題完全與資料無關，請禮貌回應並建議使用者查詢其他管道，並提醒這個是一個專注於研究所入學資訊的助理，可能無法回答其他類型的問題。
7. 若涉及到年度、名額、學分、考試方式等可能變動的資訊，請提醒使用者以最新公告為準。
8. 請用繁體中文回答所有問題。
9. 回答最後請列出「參考來源」。

使用者問題：
{question}

可參考資料：
{context}

請產生回答：
""".strip()


def generate_answer_with_openai(question: str, results: List[Dict]) -> str:
    """
    可選功能：接 OpenAI API 產生回答。

    使用前：
    1. pip install openai python-dotenv
    2. 在專案根目錄建立 .env
    3. 寫入：OPENAI_API_KEY=你的金鑰
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "尚未設定 OPENAI_API_KEY，因此目前只顯示檢索結果。"

    client = OpenAI(api_key=api_key)

    context = build_context(results)
    prompt = build_prompt(question, context)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一位可靠、謹慎、友善的研究所入學資訊助理。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def generate_answer_with_ollama(question, results, model_name="llama3.2:3b"):
    """
    使用 Ollama 本地 LLM 產生 RAG 回答。
    使用前請先：
    1. 安裝 Ollama
    2. 執行 ollama pull llama3.2:3b
    3. 確認 Ollama 正在背景執行
    """
    import ollama

    context = build_context(results)
    prompt = build_prompt(question, context)

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是一位可靠、謹慎、友善的研究所入學資訊助理。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]