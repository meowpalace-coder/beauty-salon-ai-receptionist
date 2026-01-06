# ========================================
# 粵語 AI 客服核心邏輯模組
# ========================================
# 功能：
# - 智能對話管理（狀態追蹤）
# - 「快速路由」優化（70% 對話無需 LLM）
# - 硬規則層（避免重複提問）
# - Gemini LLM 集成（複雜情況使用）

import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import google.generativeai as genai

# ==================== 環境變數載入 ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ 請先設定環境變數 GEMINI_API_KEY")

# 設定 Gemini
if os.getenv("HTTPS_PROXY"):
    genai.configure(api_key=GEMINI_API_KEY, transport="rest")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# 速度參數（從 . env 讀取，有預設值）
GEMINI_TIMEOUT_S = int(os.getenv("GEMINI_TIMEOUT_S", "5"))
GEMINI_MAX_TOKENS = int(os. getenv("GEMINI_MAX_TOKENS", "60"))
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.7"))

# ==================== 系統提示詞 ====================
VOICE_NAME = "zh-HK-HiuMaanNeural"

SYSTEM_PROMPT = """
你而家係一間「香港美容院」嘅粵語客服職員。
請用自然、親切、貼心、地道香港廣東話回應客人。

說話風格：
- 簡單易明、口語化，好似同熟客傾偈
- 有禮貌、細心，唔好成日 hard sell，感覺係幫手安排而唔係推銷
- 儘量分開短句，用多啲逗號同句號，唔好一口氣讀好長一句
- 價錢資訊可以只講一次，除非客人叫你再講多次
- 價錢用港幣（$），時間用「幾點」、「幾點半」、「AM / PM」
- 回覆入面唔好用 emoji 或奇怪符號，只用正常中文字同數字
- 回覆入面不要讀括號入面的字、唔好讀標點符號

服務範圍：
- basic facial（$480 起）
- 深層清潔 facial（$680 起）
- 皮秒激光療程（$1,800 起）
- 身體按摩（$580 起）

記憶指引：
- 一旦客人講明療程選擇，就禁止再問邊款，只可確認
- 一旦客人講明時間，就禁止再問幾時，只可確認
- 客人提供名字 / 電話後，簡單確認即可，唔好重複朗讀

回覆要求：
- 每次最多 3 句（短句為主）
- 唔好長篇介紹，直接有用訊息優先
"""

GEN_CONFIG = genai.GenerationConfig(
    max_output_tokens=GEMINI_MAX_TOKENS,
    temperature=GEMINI_TEMPERATURE,
    top_p=GEMINI_TOP_P,
)

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=SYSTEM_PROMPT,
)

# ==================== 狀態管理函數 ====================

def update_conversation_state(current_state:  dict, user_text: str) -> dict:
    """
    從客人說話中提取關鍵資訊並更新狀態。
    返回更新後的 state dict。
    """
    new_state = current_state.copy()
    u = user_text or ""

    # 🔹 療程偵測
    if not new_state.get("treatment"):
        if "深層清潔" in u or "deep" in u.lower():
            new_state["treatment"] = "深層清潔 facial"
        elif "basic facial" in u. lower() or ("basic" in u.lower() and "facial" in u.lower()):
            new_state["treatment"] = "basic facial"
        elif "皮秒" in u or "激光" in u: 
            new_state["treatment"] = "皮秒激光療程"
        elif "按摩" in u or "body" in u.lower():
            new_state["treatment"] = "身體按摩"

    # 🔹 時間偵測
    if not new_state.get("booking_time"):
        time_keywords = ["星期", "禮拜", "聽日", "後日", "今日", "今晚", "下晝", "夜晚", "點", "am", "pm"]
        if any(k in u for k in time_keywords):
            new_state["booking_time"] = u.strip()[:50]

    return new_state


def build_memory_context(state: dict) -> str:
    """
    將已知客人資料轉換為 LLM 提示詞。
    """
    items = []

    if state.get("treatment"):
        items.append(
            f"客人已經選擇療程：{state['treatment']}。"
            "唔好再問邊款療程，只可以重複確認。"
        )

    if state.get("booking_time"):
        items.append(
            f"客人預約時間：{state['booking_time']}。"
            "唔好再問幾時方便，只可以重複確認。"
        )

    if not items:
        return ""

    return (
        "【客人資料（本輪對話中已知，必須記住）】\n"
        + "\n".join(f"- {x}" for x in items)
        + "\n"
    )


# ==================== 【最高優先】快速路由規則 ====================

def _should_use_quick_path(user_text: str, state: dict) -> bool:
    """
    判斷係必要用快速規則回覆（毋須 LLM），加快速度。
    大約 70% 嘅對話會命中呢度，避免 Gemini 延遲。
    """
    u = user_text. lower()

    # 預約流程：直接規則處理（最快）
    booking_signals = ["預約", "約", "book", "改期", "幾點", "幾時", "邊日", "時間"]
    if any(sig in u for sig in booking_signals):
        return True

    # 價格查詢：預設回覆
    price_signals = ["幾錢", "幾多錢", "價錢", "費用", "cost"]
    if any(sig in u for sig in price_signals):
        return True

    # 療程介紹 + 已有療程選擇：用規則確認
    facial_signals = ["facial", "面部", "清潔", "皮膚", "做面"]
    if any(sig in u for sig in facial_signals) and state.get("treatment"):
        return True

    return False


def quick_rule_reply(user_text: str, state: dict) -> str:
    """
    【最高優先優化】用硬規則快速回覆，避免 LLM 延遲。
    返回空字串表示無法用規則處理，交由 LLM 處理。
    """
    u = user_text.lower()
    treatment = state.get("treatment")
    booking_time = state.get("booking_time")

    # ===== 預約流程規則 =====
    if any(sig in u for sig in ["預約", "約", "book", "改期"]):
        if not treatment:
            return "好呀～你想預約邊款療程呢？basic facial 定深層清潔 facial？"
        if not booking_time:
            return f"明白～你想預約 {treatment}。你想約邊日同幾點呢？"
        # 有療程 + 有時間 → 確認
        return f"好～我幫你登記：{booking_time} 做 {treatment}。麻煩留低全名同電話號碼～"

    # ===== 價格查詢規則 =====
    if any(sig in u for sig in ["幾錢", "幾多錢", "價錢", "費用"]):
        if treatment:
            if treatment == "basic facial":
                return f"good，basic facial 係 $480 起。有咩皮膚問題想重點改善嗎？"
            elif treatment == "深層清潔 facial":
                return f"deep facial 係 $680 起。幫你深層清潔同補水。"
            else:
                return f"你揀嘅 {treatment} 根據療程時間唔同，約 $680-$1800 左右。"
        else: 
            return "basic facial $480 起，深層清潔 $680，皮秒激光 $1800 起。你想了解邊款呢？"

    # ===== 療程確認規則（已有療程選擇） =====
    if any(sig in u for sig in ["facial", "清潔", "皮膚"]) and treatment:
        return f"好呀，關於 {treatment}，我哋可以幫你安排。你想幾時嚟做呢？"

    # ===== 其他常見問題 =====
    if "營業時間" in u or "幾時開" in u or "幾時收" in u:
        return "我哋營業時間係早上十一點到夜晚九點，星期一休息。"

    if "位置" in u or "地址" in u or "邊度" in u:
        return "我哋喺中環，具體地址你可以聯絡我時再畀你。你想先預約嗎？"

    return ""  # 無法用規則處理，交由 LLM


# ==================== 文本清理函數 ====================

def strip_brackets_and_symbols(text: str) -> str:
    """移除括號及符號（避免 TTS 讀出不必要內容）"""
    import re
    if not text:
        return ""
    # 移除中英文括號內容
    text = re.sub(r"[（\(][^（）\(\)]{0,40}[）\)]", "", text)
    # 移除特殊符號
    text = text.replace("*", "").replace('"', "").replace("'", "")
    text = re.sub(r"[\-─═]+", "", text)
    return text.strip()


def apply_hard_rules_to_reply(raw_reply: str, state: dict) -> str:
    """
    硬規則層：根據已知狀態過濾 LLM 回覆，避免重複提問。
    """
    if not raw_reply:
        return raw_reply

    sentences = re.split(r"(? <=[。！？\?! ])\s*", raw_reply)
    filtered = []

    has_treatment = state.get("treatment") is not None
    has_time = state.get("booking_time") is not None

    for s in sentences:
        if not s.strip():
            continue

        skip = False

        # 已有療程 → 禁止再問療程
        if has_treatment and any(kw in s for kw in ["想做咩", "邊款療程", "做邊款", "邊隻 facial"]):
            skip = True

        # 已有時間 → 禁止再問時間
        if has_time and any(kw in s for kw in ["幾點", "幾時", "邊日", "咩時間"]):
            skip = True

        if not skip:
            filtered.append(s. strip())

    cleaned = "。".join(filtered).strip()
    if not cleaned. endswith("。") and cleaned:
        cleaned += "。"

    return strip_brackets_and_symbols(cleaned)


def _extract_text_from_response(response) -> str:
    """安全抽取 Gemini 回應"""
    try:
        if hasattr(response, "text") and response.text:
            return response.text. strip()
    except Exception:
        pass

    try:
        for cand in getattr(response, "candidates", []):
            for part in getattr(cand, "content", {}).parts: 
                if hasattr(part, "text") and part.text:
                    return part.text.strip()
    except Exception:
        pass

    return ""


# ==================== 【主函數】generate_reply ====================

def generate_reply(user_text: str, current_state: dict) -> tuple[str, dict]:
    """
    主函數：根據用戶輸入生成回覆。
    返回 (回覆文字, 更新後嘅狀態)
    
    流程：
    1. 更新狀態
    2. 嘗試快速路由（70% 情況）
    3. 若快速路由無效，才問 LLM
    """
    if not user_text:
        return "唔好意思，我頭先好似聽唔清楚，可以再講多次嗎？", current_state

    # 🔹 第一步：更新狀態
    new_state = update_conversation_state(current_state, user_text)

    # 🔹 第二步：嘗試快速路由（最快，無 LLM 延遲）
    if _should_use_quick_path(user_text, new_state):
        fast_reply = quick_rule_reply(user_text, new_state)
        if fast_reply:
            return fast_reply, new_state

    # 🔹 第三步：若快速路由無效，問 LLM（複雜對話）
    try:
        def _call_gemini():
            memory_ctx = build_memory_context(new_state)
            prompt = (
                f"{memory_ctx}"
                f"客人：「{user_text}」\n你："
            )
            return gemini_model.generate_content(
                prompt,
                generation_config=GEN_CONFIG,
                request_options={"timeout":  GEMINI_TIMEOUT_S},
            )

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call_gemini)
            response = fut.result(timeout=GEMINI_TIMEOUT_S)

        reply_text = _extract_text_from_response(response)

        if reply_text: 
            # 套用硬規則過濾
            reply_text = apply_hard_rules_to_reply(reply_text, new_state)
            return reply_text, new_state

        return "唔好意思，我暫時回應唔到，可以重新講嗎？", new_state

    except FuturesTimeoutError:
        # LLM 超時 → 用快速後備回覆
        fallback = quick_rule_reply(user_text, new_state)
        return fallback or "系統暫時繁忙，可以稍後再試嗎？", new_state

    except Exception as e:
        print(f"❌ LLM 錯誤：{e}")
        fallback = quick_rule_reply(user_text, new_state)
        return fallback or "唔好意思，出咗啲技術問題，可以再講一次嗎？", new_state


# ==================== 【輔助函數】重置記憶 ====================

def reset_memory() -> dict:
    """重置對話狀態（供 Flask /api/reset 端點使用）"""
    return {
        "treatment": None,
        "booking_time": None,
    }


# ==================== 【輔助函數】文本淨化 ====================

def sanitize_tts_text(text: str) -> str:
    """清理要交俾 TTS 嘅文字"""
    if not text:
        return ""
    text = strip_brackets_and_symbols(text)
    text = re.sub(r"(斜線|slash|／|\\|/)+", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text