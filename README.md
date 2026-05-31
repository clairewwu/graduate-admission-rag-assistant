# 研究所入學 RAG 問答小助理

本專案是一個以研究所入學資訊為應用場景的 RAG 問答系統，目標是協助準研究生與新生快速查詢系所相關資訊，例如教師研究方向、課程資訊、組別介紹、入學資訊與畢業規則。

系所資訊通常分散在招生簡章、教師介紹頁、課程規劃表、研究生手冊與系所公告中，使用者需要花時間在不同文件之間查找。本專案透過語意檢索與本地 LLM 回答生成，建立一個可追溯來源的入學資訊小助理。

---

## 專案功能

* 支援自然語言查詢系所資訊
* 使用 SentenceTransformer 將系所資料轉為 embedding
* 使用 FAISS 建立向量索引並進行語意檢索
* 支援短查詢 keyword-first search，例如「口試」、「學分」、「老師」（或直接是教授名字）
* 支援 Ollama 本地 LLM 根據檢索資料生成回答
* 顯示實際參考資料片段，包含標題、類別、內容與來源
* 加入 unsupported intent detection，避免回答資料庫未收錄的問題
* 加入 category coverage check，降低 LLM 在資料不足時產生幻覺
---

## 技術架構

```text
使用者問題
    ↓
Guardrails 檢查
    ↓
短查詢 keyword search / 一般查詢 embedding search
    ↓
FAISS 檢索相關 chunks
    ↓
組成 RAG prompt
    ↓
Ollama 本地 LLM 生成回答
    ↓
Streamlit 顯示回答與參考資料
```

---

## 使用技術

* Python
* Streamlit
* pandas
* SentenceTransformers
* FAISS
* Ollama
* Llama / Qwen 本地模型
* CSV-based knowledge base

---

## 資料格式

本專案使用 CSV 作為初版知識庫格式。每一列資料代表一個語意完整的 chunk。

```csv
id,category,title,content,source,url
1,教師資訊,XXX老師研究方向,XXX老師為語言所的講座教授及人社中心主任，研究領域包含語言與認知、認知語法、手語語言學、語用學、漢語結構、華語第二語言習得與高齡者語言與認知能力。適合對認知科學以及高齡語言老化等的學生參考。,教師介紹頁,http://.......
```

欄位說明：

| 欄位       | 說明                    |
| -------- | --------------------- |
| id       | 資料編號                  |
| category | 資料類型，例如教師資訊、課程資訊、畢業規則 |
| degree   | 學制，例如碩士班、博士班、共同、未指定   |
| title    | chunk 標題              |
| content  | 實際檢索與回答依據             |
| source   | 資料來源                  |
| url      | 原始網址                  |

---

## RAG 流程

1. 將系所資料整理為 CSV，每列作為一個 chunk。
2. 使用 SentenceTransformer 將 chunk 轉換為 embedding。
3. 使用 FAISS 建立向量索引。
4. 使用者輸入問題後，系統先進行 guardrails 檢查。
5. 若問題長度小於等於 3，優先使用 keyword search。
6. 若不是短查詢，則使用 embedding search 找出相關 chunks。
7. 將檢索結果組成 prompt，交由 Ollama 本地 LLM 生成回答。
8. 介面顯示 LLM 回答與實際參考資料片段。

---

## Guardrails 防幻覺設計

RAG 系統常見問題是：即使資料庫沒有答案，LLM 仍可能根據相似但不具回答性的資料產生幻覺。為降低此問題，本專案加入以下設計：

### 1. Unsupported intent detection

針對目前資料庫未收錄或高度容易幻覺的問題類型，系統會在進入 LLM 前直接拒答。例如：

* 知名校友
* 薪資與就業數據
* 即時收生名額
* 生活與校外資訊

### 2. Category coverage check

系統會根據使用者問題推測所需資料類別，例如教師資訊、課程資訊、畢業規則等，並檢查知識庫是否包含該類別。若資料庫沒有對應 category，系統不會進入生成階段。

### 3. Short query keyword-first search

短查詢如「口試」、「學分」、「老師」因語境不足，使用 embedding search 時可能不穩定。因此系統會先進行 keyword search，若無結果再退回 embedding search。

### 4. Grounded prompt

LLM prompt 明確要求模型只能根據可參考資料回答，不得自行編造老師、課程、校友、學分、年份、頁碼或來源。

---

## 安裝方式

```bash
pip install -r requirements.txt
```

若使用 Ollama 本地模型，請先安裝 Ollama，並下載模型：

```bash
ollama pull llama3.2:3b
```

或：

```bash
ollama pull qwen3:4b
```

---

## 執行方式

第一次使用或更新 CSV 後，請先建立向量索引：

```bash
python build_index.py
```

啟動 Streamlit：

```bash
python -m streamlit run app.py
```

---

## 已知限制

* 本專案目前以人工整理 CSV 作為知識庫，尚未加入 PDF 自動解析流程。
* 本地小型 LLM 可能仍會出現不穩定回答，因此系統加入 guardrails 與 grounded prompt 降低風險。
* 目前檢索主要使用 FAISS embedding search 與 keyword search，尚未加入完整 BM25 hybrid search 或 reranker。
* 若資料庫未收錄某類資訊，系統會拒答，而不是根據相似資料推論。
* 招生、課程與畢業規則可能隨學年度更新，實際資訊仍應以系所官網與最新公告為準。

---

## Future Work

* 加入 PDF parsing，自動處理招生簡章與研究生手冊
* 建立 evaluation questions，計算 Recall與 no-answer accuracy
* 改善 metadata filtering，支援學制、組別、學年度等條件
* 部署線上 retrieval demo，並保留本地 Ollama LLM 模式
* 針對碩士班與博士班資訊加入 degree 欄位，降低學制混淆風險

---

## 專案定位

本專案是一個以 retrieval reliability 為核心的 RAG prototype。重點在於整合分散的系所資訊、建立可檢索知識庫、降低資料不足時的幻覺回答，並讓使用者能追溯回答依據。
