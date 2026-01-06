# ========================================
# Flask 後端應用程式
# ========================================
# 功能：
# - 前端音訊上傳接收
# - Azure STT 語音轉文字
# - Gemini LLM 對話
# - Azure TTS 文字轉語音
# - 日誌追蹤

import os
import uuid
import time
import traceback
import json
import subprocess
import shutil
from pathlib import Path
from threading import Lock
import logging

from flask import Flask, request, jsonify, send_file, abort, Response
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# 載入環境變數
load_dotenv()

# 引入核心邏輯
import core_logic

# ==================== 日誌設定 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('beauty_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== Flask 應用設定 ====================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "5001"))

# ==================== Azure 語音設定 ====================
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastasia")

if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
    raise RuntimeError("❌ 請設定 AZURE_SPEECH_KEY 及 AZURE_SPEECH_REGION")

VOICE_NAME = getattr(core_logic, "VOICE_NAME", "zh-HK-HiuMaanNeural")

# ==================== 臨時目錄設定 ====================
TMP_DIR = Path(os.getenv("FLASK_TMP_DIR", "./flask_tmp"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 語音配置（STT / TTS 分離） ====================
speech_config_stt = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
speech_config_stt.speech_recognition_language = "zh-HK"

speech_config_tts = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
speech_config_tts.speech_synthesis_language = "zh-HK"
speech_config_tts.speech_synthesis_voice_name = VOICE_NAME
speech_config_tts.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
)

_TTS_LOCK = Lock()

# ==================== 工具函數 ====================

def _now():
    """取得當前時間戳（秒），用於計時"""
    return time.perf_counter()


def _ensure_ffmpeg():
    """檢查 ffmpeg 係否安裝"""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("❌ ffmpeg 未安裝或未加入 PATH")


def convert_to_wav(src:  Path, dst: Path):
    """用 ffmpeg 將 webm 轉成 wav（16kHz, mono, PCM）"""
    _ensure_ffmpeg()
    cmd = [
        "ffmpeg", "-y",
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-threads", "2",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dst)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 轉換失敗：{p.stderr[-500:] if p.stderr else '未知錯誤'}")


def stt_from_wav(wav_path: Path) -> str:
    """Azure STT：將 wav 檔轉換為廣東話文字"""
    try:
        audio_cfg = speechsdk.audio.AudioConfig(filename=str(wav_path))
        recog = speechsdk.SpeechRecognizer(speech_config_stt, audio_cfg)
        r = recog.recognize_once_async().get()

        if r.reason == speechsdk.ResultReason.RecognizedSpeech:
            return (r.text or "").strip()

        if r.reason == speechsdk. ResultReason.NoMatch:
            logger.warning("STT：聽唔到說話")
            return ""

        if r.reason == speechsdk.ResultReason.Canceled:
            logger.error(f"STT 被取消：{r.cancellation_details. error_details}")
            return ""

        return ""
    except Exception as e: 
        logger.error(f"STT 異常：{e}", exc_info=True)
        return ""


def tts_to_mp3(text: str, out:  Path) -> tuple[bool, str]:
    """
    Azure TTS：將文字轉成 MP3 語音檔
    返回 (成功否, 錯誤訊息)
    """
    # 清理文字
    t = core_logic.sanitize_tts_text(text) if hasattr(core_logic, 'sanitize_tts_text') else (text or "")
    t = t.strip()

    if not t:
        return False, "文字為空"

    # 限制長度（加快 TTS）
    max_chars = int(os.getenv("MAX_REPLY_CHARS", "180"))
    if len(t) > max_chars:
        t = t[:max_chars]

    try:
        if out.exists():
            out.unlink()
    except Exception: 
        pass

    try: 
        audio_cfg = speechsdk.audio.AudioOutputConfig(filename=str(out))
        synth = speechsdk.SpeechSynthesizer(speech_config_tts, audio_cfg)

        with _TTS_LOCK:
            r = synth.speak_text_async(t).get()

        if r.reason == speechsdk.ResultReason.SynthesizingAudioCompleted and out.exists() and out.stat().st_size > 0:
            return True, ""

        if r.reason == speechsdk.ResultReason.Canceled:
            err = r.cancellation_details. error_details if r.cancellation_details else "未知"
            logger.error(f"TTS 被取消：{err}")
            return False, f"TTS 失敗：{err}"

        logger.error(f"TTS 未完成：{r.reason}")
        return False, f"TTS 失敗：{r.reason}"

    except Exception as e:
        logger.error(f"TTS 異常：{e}", exc_info=True)
        return False, str(e)


def _safe_delete(p: Path):
    """安全刪除檔案"""
    try:
        if p and p.exists():
            p.unlink()
    except Exception:
        pass


# ==================== API 端點 ====================

@app.post("/api/voice")
def api_voice():
    """主 API：接收音訊、轉錄、生成回覆、合成語音"""
    request_id = uuid.uuid4().hex[: 8]
    logger.info(f"[{request_id}] 新請求開始")

    t0 = _now()
    webm_path = wav_path = None

    try: 
        # 1️⃣ 取得上傳嘅音訊檔
        f = request.files. get("audio")
        if not f or not f.filename:
            logger.warning(f"[{request_id}] 無音訊檔")
            return jsonify(ok=False, error="沒有上傳音訊檔"), 400

        # 2️⃣ 取得對話狀態
        state_str = request.form.get("state", "{}")
        try:
            current_state = json.loads(state_str)
        except json. JSONDecodeError:
            current_state = {}

        # 3️⃣ 儲存上傳嘅檔案並轉換
        rid = uuid.uuid4().hex
        webm_path = TMP_DIR / f"in_{rid}.webm"
        wav_path = TMP_DIR / f"in_{rid}.wav"

        f.save(webm_path)
        logger.info(f"[{request_id}] 檔案已儲存：{webm_path. name}")

        t_a = _now()
        convert_to_wav(webm_path, wav_path)
        t_conv = _now() - t_a
        logger.info(f"[{request_id}] 轉換耗時：{t_conv:. 2f}s")

        # 4️⃣ 語音轉文字 (STT)
        t_b = _now()
        user_text = stt_from_wav(wav_path)
        t_stt = _now() - t_b

        if not user_text:
            logger.warning(f"[{request_id}] STT 無結果")
            return jsonify(
                ok=False,
                error="無法識別語音",
                state=current_state,
                timing=f"{_now()-t0:.1f}s"
            ), 400

        logger.info(f"[{request_id}] STT 完成：'{user_text[: 50]}'...  (耗時 {t_stt:.2f}s)")

        # 5️⃣ 生成 AI 回覆
        t_c = _now()
        reply_text, new_state = core_logic.generate_reply(user_text, current_state)
        t_llm = _now() - t_c

        if not reply_text:
            reply_text = "唔好意思，我頭先聽唔清楚，可以再講一次嗎？"

        logger.info(f"[{request_id}] AI 回覆耗時：{t_llm:.2f}s，回覆：'{reply_text[:50]}'...")

        # 6️⃣ 文字轉語音 (TTS)
        mp3_path = TMP_DIR / f"tts_{rid}.mp3"
        t_d = _now()
        tts_ok, tts_err = tts_to_mp3(reply_text, mp3_path)
        t_tts = _now() - t_d

        logger.info(f"[{request_id}] TTS 耗時：{t_tts:.2f}s，成功：{tts_ok}")

        # 7️⃣ 回傳結果
        total_time = _now() - t0
        logger.info(f"[{request_id}] 總耗時：{total_time:.2f}s")

        return jsonify(
            ok=True,
            tts_ok=tts_ok,
            tts_error=tts_err,
            user_text=user_text,
            reply_text=reply_text,
            audio_url=(f"/tts/{mp3_path.name}" if tts_ok else ""),
            state=new_state,
            timing=f"{total_time:.1f}s"
        )

    except Exception as e: 
        logger.error(f"[{request_id}] 異常：{str(e)}", exc_info=True)
        state_str = request.form.get("state", "{}")
        try:
            current_state = json.loads(state_str)
        except json.JSONDecodeError:
            current_state = {}
        return jsonify(ok=False, error=str(e), state=current_state), 500

    finally:
        _safe_delete(webm_path)
        _safe_delete(wav_path)


@app.post("/api/reset")
def api_reset():
    """重置對話記憶"""
    try:
        new_state = core_logic.reset_memory()
        logger.info("對話記憶已重置")
        return jsonify(ok=True, state=new_state)
    except Exception as e: 
        logger.error(f"重置記憶失敗：{e}")
        return jsonify(ok=False, error=str(e)), 500


@app.get("/tts/<name>")
def tts_file(name):
    """播放 TTS 語音檔"""
    if not name.startswith("tts_") or not name.endswith(".mp3"):
        abort(404)
    p = TMP_DIR / name
    if not p.is_file():
        abort(404)
    return send_file(p, mimetype="audio/mpeg", conditional=True)


# ==================== 前端 HTML ====================

HTML_PAGE = r"""<! doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>美容院 粵語 AI 接線生 Demo</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans HK", "PingFang HK", Arial, sans-serif;
      max-width: 800px;
      margin: 0 auto;
      padding: 20px;
      background:  linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
    }
    .container {
      background: white;
      border-radius: 16px;
      padding: 24px;
      box-shadow:  0 20px 60px rgba(0,0,0,0.3);
    }
    h1 {
      color: #d63384;
      margin:  0 0 8px;
      font-size: 36px;
    }
    .subtitle {
      color: #666;
      margin: 0 0 20px;
      font-size:  14px;
    }
    . feature-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 20px;
    }
    . feature-card {
      background:  #f0f8ff;
      border-left: 4px solid #28a745;
      padding: 10px 12px;
      border-radius:  6px;
      font-size: 12px;
      color: #333;
    }
    .controls {
      display: flex;
      gap: 8px;
      margin-bottom:  20px;
      flex-wrap: wrap;
    }
    button {
      padding: 12px 16px;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s;
    }
    #btnStart {
      background: #28a745;
      color: white;
    }
    #btnStart:hover: not(:disabled) {
      background: #218838;
    }
    #btnStop {
      background: #dc3545;
      color:  white;
    }
    #btnStop:hover:not(:disabled) {
      background: #c82333;
    }
    #btnReset {
      background: #6c757d;
      color: white;
    }
    #btnReset:hover:not(:disabled) {
      background: #5a6268;
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .status-bar {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-bottom: 20px;
      padding: 12px;
      background: #f8f9fa;
      border-radius: 8px;
      font-size: 13px;
    }
    #status {
      font-weight: 600;
      color: #333;
      flex:  1;
    }
    #timing {
      font-family: monospace;
      font-weight: bold;
      color: #dc3545;
      display: none;
    }
    #timing.active {
      display: inline;
    }
    .content-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }
    .content-box {
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      padding: 12px;
    }
    .content-box h3 {
      margin:  0 0 8px;
      font-size: 13px;
      color: #666;
      font-weight: 600;
    }
    .content-box . text {
      min-height: 60px;
      padding: 8px;
      background: #fafafa;
      border-radius: 6px;
      font-size:  13px;
      line-height: 1.5;
      word-wrap: break-word;
    }
    .audio-section {
      margin-bottom: 20px;
    }
    .audio-section h3 {
      margin: 0 0 8px;
      font-size: 13px;
      color: #666;
      font-weight: 600;
    }
    audio {
      width: 100%;
      height: 32px;
    }
    #log {
      background: #1e1e1e;
      color:  #d7ffd7;
      padding: 12px;
      border-radius:  8px;
      font-family: "Courier New", monospace;
      font-size: 11px;
      max-height: 200px;
      overflow-y:  auto;
      white-space: pre-wrap;
      word-break: break-all;
      line-height: 1.4;
    }
    @media (max-width: 600px) {
      .content-grid {
        grid-template-columns: 1fr;
      }
      .feature-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🙋‍♀️ 粵語 AI 接線生 Demo</h1>
    <p class="subtitle">美容院版本 v1.0 - 自動處理預約、查詢、介紹</p>

    <div class="feature-grid">
      <div class="feature-card">✅ 粵語自然對話</div>
      <div class="feature-card">✅ <10秒回應</div>
      <div class="feature-card">✅ 智能記憶客戶</div>
      <div class="feature-card">✅ 避免重複提問</div>
    </div>

    <div class="controls">
      <button id="btnStart">🎙️ 開始錄音</button>
      <button id="btnStop" disabled>⏹️ 停止並送出</button>
      <button id="btnReset">🧼 清空記憶</button>
    </div>

    <div class="status-bar">
      <span id="status">準備就緒</span>
      <span id="timing">⏱️ 0. 0s</span>
    </div>

    <div class="content-grid">
      <div class="content-box">
        <h3>👤 你講嘅：</h3>
        <div id="sttText" class="text"></div>
      </div>
      <div class="content-box">
        <h3>🤖 接線生回覆：</h3>
        <div id="replyText" class="text"></div>
      </div>
    </div>

    <div class="audio-section">
      <h3>🔊 語音回覆：</h3>
      <audio id="player" controls></audio>
    </div>

    <div style="margin-top: 16px;">
      <h3 style="font-size: 13px; color: #666; margin:  0 0 8px;">📋 系統日誌：</h3>
      <pre id="log"></pre>
    </div>
  </div>

<script>
const $ = (id) => document.getElementById(id);

let stream = null;
let mediaRecorder = null;
let chunks = [];
let conversationState = {};
let recordStartTime = 0;

function log(msg) {
  const now = new Date().toLocaleTimeString();
  $("log").textContent = `[${now}] ${msg}\n` + $("log").textContent;
}

function pickMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg"
  ];
  for (const t of candidates) {
    if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

async function initMic() {
  try {
    if (!(window.isSecureContext || location.hostname === "localhost" || location.hostname === "127.0.0.1")) {
      $("status").textContent = "⚠️ 需要 HTTPS 或 localhost";
      log("Not secure context");
      return;
    }
    stream = await navigator.mediaDevices. getUserMedia({ audio: true });
    $("status").textContent = "✅ 麥克風就緒";
    $("btnStart").disabled = false;
    log("✅ 麥克風已授權");
  } catch (e) {
    $("status").textContent = "❌ 麥克風授權失敗";
    log("❌ 麥克風錯誤：" + e);
  }
}

initMic();

$("btnStart").onclick = async () => {
  try {
    if (!stream) {
      await initMic();
      if (!stream) return;
    }
    chunks = [];
    const mimeType = pickMimeType();
    const opts = { audioBitsPerSecond: 64000 };
    if (mimeType) opts.mimeType = mimeType;

    mediaRecorder = new MediaRecorder(stream, opts);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };

    mediaRecorder. onstart = () => {
      recordStartTime = Date.now();
      $("status").textContent = "🔴 錄音中…";
      $("btnStart").disabled = true;
      $("btnStop").disabled = false;
      log("▶️ 錄音開始");
    };

    mediaRecorder.onstop = async () => {
      $("status").textContent = "⏳ 處理中，請稍候…";
      $("btnStop").disabled = true;
      $("timing").classList.add("active");

      const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
      log(`📦 錄音完成，大小：${(blob.size / 1024).toFixed(1)} KB`);

      const fd = new FormData();
      fd.append("audio", blob, "recording.webm");
      fd.append("state", JSON.stringify(conversationState));

      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 30000);

        const res = await fetch("/api/voice", {
          method: "POST",
          body: fd,
          signal: controller.signal
        });

        clearTimeout(timer);

        const j = await res.json();

        if (j.state) {
          conversationState = j.state;
        }

        if (! j.ok) {
          $("status").textContent = `❌ 失敗：${j.error}`;
          log("❌ API 錯誤：" + (j.error || "未知"));
          $("btnStart").disabled = false;
          $("timing").classList.remove("active");
          return;
        }

        $("sttText").textContent = j.user_text || "（未能識別）";
        $("replyText").textContent = j.reply_text || "";
        $("player").src = j.audio_url || "";
        $("status").textContent = `✅ 完成 (${j.timing})`;
        log(`✅ 總耗時：${j.timing}`);
        log(`📝 客人：${j.user_text || "（無）"}`);
        log(`🤖 回覆：${j.reply_text || "（無）"}`);

        try {
          await $("player").play();
          log("🔊 正在播放語音");
        } catch (e) {
          log("⚠️ 播放失敗：" + e);
        }

        $("btnStart").disabled = false;
        $("timing").classList.remove("active");

      } catch (e) {
        $("status").textContent = "❌ 上傳失敗";
        log("❌ 上傳錯誤：" + e);
        $("btnStart").disabled = false;
        $("timing").classList.remove("active");
      }
    };

    mediaRecorder.start();

  } catch (e) {
    $("status").textContent = "❌ 錄音失敗";
    log("❌ 錄音錯誤：" + e);
    $("btnStart").disabled = false;
  }
};

$("btnStop").onclick = () => {
  try {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder. stop();
      log("⏹️ 手動停止錄音");
    }
  } catch (e) {
    log("❌ 停止錯誤：" + e);
  }
};

$("btnReset").onclick = async () => {
  try {
    const r = await fetch("/api/reset", { method:  "POST" });
    const j = await r.json();
    if (j.ok) {
      conversationState = j.state || {};
      $("sttText").textContent = "";
      $("replyText").textContent = "";
      $("player").src = "";
      log("🧼 對話記憶已清空");
      $("status").textContent = "✅ 記憶已清空，可以開始新對話";
    } else {
      log("❌ 清空失敗：" + (j.error || "未知錯誤"));
    }
  } catch (e) {
    log("❌ 清空錯誤：" + e);
  }
};

// 即時計時顯示（用於 UX 反饋）
setInterval(() => {
  if ($("status").textContent.includes("處理中")) {
    let elapsed = ((Date.now() - recordStartTime) / 1000).toFixed(1);
    $("timing").textContent = `⏱️ ${elapsed}s`;
  }
}, 100);
</script>
</body>
</html>
"""

@app.get("/")
def index():
    """主頁面"""
    return Response(HTML_PAGE, mimetype="text/html")


if __name__ == "__main__": 
    try:
        _ensure_ffmpeg()
        logger.info("✅ ffmpeg 檢查通過")
        logger.info(f"🚀 Flask 應用啟動：http://{HOST}:{PORT}")
        logger.info(f"📍 Azure 區域：{AZURE_SPEECH_REGION}")
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    except RuntimeError as e:
        logger. error(f"❌ 啟動錯誤：{e}")