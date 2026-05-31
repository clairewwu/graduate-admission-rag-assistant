# src/guardrails.py

from typing import Optional, Dict, List, Set


# 目前系統明確不支援，或高度容易幻覺的問題類型
UNSUPPORTED_INTENTS = {
    "校友資訊": [
        "校友", "系友", "畢業生", "有名的人", "名人",
        "傑出校友", "知名校友", "學長姐", "學長", "學姊"
    ],
    "薪資與就業數據": [
        "薪水", "年薪", "月薪", "待遇", "薪資",
        "就業率", "錄取率", "出路排名"
    ],
    "即時名額或收生狀況": [
        "今年收不收", "今年會收", "收幾個學生",
        "有沒有名額", "還有沒有缺", "老師今年收"
    ],
    "生活與校外資訊": [
        "美食", "餐廳", "便當", "咖啡廳", "滷肉飯",
        "租屋", "房租", "宿舍可以養", "養貓", "養狗",
        "天氣", "景點", "旅遊"
    ],
}


# 問題需要哪種資料類別
REQUIRED_CATEGORY_RULES = {
    "校友資訊": [
        "校友", "系友", "畢業生", "有名的人", "名人",
        "傑出校友", "知名校友", "學長姐", "學長", "學姊"
    ],
    "教師資訊": [
        "老師", "教授", "指導教授", "找誰", "誰研究",
        "研究方向", "專長", "實驗室"
    ],
    # "課程資訊": [
    #     "課", "課程", "修課", "必修", "選修",
    #     "可以修", "開課", "課表"
    # ],
    "組別資訊": [
        "組", "組別", "分組", "哪一組", "有哪幾組",
        "一般語言學組", "華語教學組", "手語語言學組"
    ],
    "入學資訊": [
        "入學", "申請", "報考", "備審", "招生",
        "口試準備", "考試", "推甄", "甄試"
    ],
    "畢業規則": [
        "畢業", "畢業學分", "畢業口試", "論文", "離校",
        "畢業門檻", "修業年限"
    ],
    "新生資訊": [
        "新生", "報到", "選課建議", "找研究方向",
        "剛入學", "入學後"
    ],
}


def detect_unsupported_intent(query: str) -> Optional[str]:
    """
    判斷使用者問題是否屬於目前系統不支援或高度容易幻覺的意圖。
    若命中，回傳 intent 名稱；否則回傳 None。
    """
    for intent, keywords in UNSUPPORTED_INTENTS.items():
        if any(keyword in query for keyword in keywords):
            return intent

    return None


def infer_required_category(query: str) -> Optional[str]:
    """
    根據問題中的關鍵詞，推測這題需要哪一種資料類別。
    若無法判斷，回傳 None。
    """
    for category, keywords in REQUIRED_CATEGORY_RULES.items():
        if any(keyword in query for keyword in keywords):
            return category

    return None


def get_available_categories(metadata: List[Dict]) -> Set[str]:
    """
    從 metadata 中取得目前資料庫實際有哪些 category。
    """
    return set(item["category"] for item in metadata)


def check_category_coverage(query: str, metadata: List[Dict]) -> Dict:
    """
    檢查：
    1. 問題是否屬於不支援意圖
    2. 問題需要的 category 是否存在於資料庫

    回傳格式：
    {
        "can_answer": bool,
        "reason": str,
        "required_category": str | None,
        "unsupported_intent": str | None
    }
    """
    unsupported_intent = detect_unsupported_intent(query)

    # 若明確是不支援意圖，先擋掉。
    if unsupported_intent:
        return {
            "can_answer": False,
            "reason": (
                f"目前資料庫沒有收錄「{unsupported_intent}」相關資訊，"
                "因此無法根據資料回答。建議查詢系所官網或詢問系辦。"
            ),
            "required_category": None,
            "unsupported_intent": unsupported_intent,
        }

    required_category = infer_required_category(query)
    available_categories = get_available_categories(metadata)

    # 若推測出需要的 category，但資料庫沒有這類資料，就拒答。
    if required_category and required_category not in available_categories:
        return {
            "can_answer": False,
            "reason": (
                f"這個問題需要「{required_category}」相關資料，"
                f"但目前資料庫只有：{', '.join(sorted(available_categories))}。"
                "因此無法根據資料回答，建議查詢系所官網或詢問系辦。"
            ),
            "required_category": required_category,
            "unsupported_intent": None,
        }

    return {
        "can_answer": True,
        "reason": "pass",
        "required_category": required_category,
        "unsupported_intent": None,
    }