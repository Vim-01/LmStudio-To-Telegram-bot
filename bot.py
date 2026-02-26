import os
import json
import logging
import asyncio
import base64
import re
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Telegram
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler,
)
from telegram.request import HTTPXRequest
from telegram.error import BadRequest

# HTTP
import httpx

# Загрузка конфигурации
load_dotenv()

# --- LOGGING ---
logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot_debug.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIG FROM .ENV ---
CFG = {
    "TOKEN": os.getenv("TELEGRAM_TOKEN", ""),
    "ADMIN_ID": int(os.getenv("ADMIN_ID", "0")),
    "BASE_URL": os.getenv("LM_STUDIO_URL", "http://localhost:1234").rstrip('/'),
    "API_KEY": os.getenv("LM_API_TOKEN", "lm-studio"),
    "BROWSER_TTL": int(os.getenv("BROWSER_TTL", "600")),
    "TIMEOUT": float(os.getenv("TIMEOUT", "300.0")),
    "MAX_HISTORY": 40,
    "STREAM_UPDATE_INTERVAL": 1.5,  # секунды между обновлениями (защита от FloodWait)
    "ENABLE_VISION": os.getenv("ENABLE_VISION", "true").lower() == "true",
    "ENABLE_STREAMING": os.getenv("ENABLE_STREAMING", "true").lower() == "true",
}

# Валидация критичных параметров
if not CFG["TOKEN"]:
    logger.error("❌ TELEGRAM_TOKEN не задан в .env")
    exit(1)
if CFG["ADMIN_ID"] == 0:
    logger.error("❌ ADMIN_ID не задан в .env")
    exit(1)

# --- CONFIG FROM config.json ---
def load_config_json() -> Dict:
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load config.json: {e}")
        return {"models": {}, "prompts": {}, "ui": {"keyboard": []}}

CONFIG_JSON = load_config_json()
MODELS = CONFIG_JSON.get("models", {})
PROMPTS = CONFIG_JSON.get("prompts", {})
KEYBOARD = CONFIG_JSON.get("ui", {}).get("keyboard", [])

# Эндпоинты API
API_URL_NATIVE = f"{CFG['BASE_URL']}/api/v1"
API_URL_OAI = f"{CFG['BASE_URL']}/v1"

# Глобальные переменные
last_browser_usage = 0
sessions: Dict[str, 'ChatSession'] = {}
current_chat = "Main"
SESSIONS_FILE = "sessions.json"
bot_instance = None  # Для редактирования сообщений

# --- SESSION CLASS ---
class ChatSession:
    def __init__(self, name: str = "Main", mode: str = "sophia"):
        self.name = name
        self.mode = mode
        self.messages: List[Dict[str, str]] = []
        self.custom_rp: Optional[str] = None
        self.show_thinking = False
        self.awaiting_rp_input = False
        self.last_message_id: Optional[int] = None
        self.last_chat_id: Optional[int] = None
        self.last_update_time: float = 0

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > CFG["MAX_HISTORY"]:
            self.messages = self.messages[-CFG["MAX_HISTORY"]:]

    def get_context(self, new_input: Optional[str] = None, vision_context: str = "") -> List[Dict[str, str]]:
        sys_prompt = PROMPTS.get(self.mode, PROMPTS.get("sophia", "You are a helpful assistant."))
        if self.custom_rp:
            sys_prompt += f"\n[SCENARIO: {self.custom_rp}]"
        sys_prompt += f"\n[System Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}]"
        ctx = [{"role": "system", "content": sys_prompt}]
        ctx.extend(self.messages)
        if new_input:
            final_user_msg = f"{vision_context}\n{new_input}" if vision_context else new_input
            ctx.append({"role": "user", "content": final_user_msg})
        return ctx

# --- PERSISTENCE ---
def save_sessions():
    try:
        data = {}
        for k, v in sessions.items():
            data[k] = {
                "name": v.name,
                "mode": v.mode,
                "msgs": v.messages,
                "rp": v.custom_rp,
                "thinking": v.show_thinking
            }
        data["current"] = current_chat
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Сессии сохранены ({len(sessions)} сессий)")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения сессий: {e}")

def load_sessions():
    global sessions, current_chat
    try:
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            current_chat = d.pop("current", "Main")
            for k, v in d.items():
                s = ChatSession(v["name"], v["mode"])
                s.messages = v.get("msgs", [])
                s.custom_rp = v.get("rp")
                s.show_thinking = v.get("thinking", False)
                sessions[k] = s
            logger.info(f"📂 Загружено {len(sessions)} сессий, текущая: {current_chat}")
        if not sessions:
            sessions["Main"] = ChatSession()
            logger.info("🆕 Создана новая сессия 'Main'")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки сессий: {e}")
        sessions["Main"] = ChatSession()

# --- MODEL MANAGEMENT (Надёжная реализация) ---
async def get_full_model_id(client: httpx.AsyncClient, search_name: str, headers: Dict) -> str:
    """Находит полный путь модели в LM Studio по частичному совпадению."""
    try:
        r = await client.get(f"{API_URL_NATIVE}/models", headers=headers, timeout=10.0)
        if r.status_code != 200:
            logger.error(f"❌ Ошибка получения списка моделей: {r.status_code}")
            return search_name
        data = r.json()
        models = data.get("models", [])
        # Точное совпадение
        for m in models:
            m_path = m.get("path") or m.get("id") or m.get("key", "")
            if search_name.lower() == m_path.lower():
                return m_path
        # Частичное совпадение
        for m in models:
            m_path = m.get("path") or m.get("id") or m.get("key", "")
            if search_name.lower() in m_path.lower():
                return m_path
        logger.warning(f"⚠️ Модель '{search_name}' не найдена, используем как есть")
        return search_name
    except Exception as e:
        logger.error(f"❌ Ошибка поиска модели: {e}")
        return search_name
async def ensure_model_loaded(mode_key: str) -> Optional[str]:
    """
    Надёжная загрузка модели с выгрузкой других инстансов.
    Возвращает полный путь модели или None при ошибке.
    """
    short_name = MODELS.get(mode_key)
    if not short_name:
        logger.error(f"❌ Модель для режима '{mode_key}' не задана в config.json")
        return None

    headers = {
        "Authorization": f"Bearer {CFG['API_KEY']}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            full_id = await get_full_model_id(client, short_name, headers)
            logger.info(f"🔍 Целевая модель: {full_id}")

            r = await client.get(f"{API_URL_NATIVE}/models", headers=headers, timeout=10.0)
            if r.status_code != 200:
                logger.error(f"❌ Не удалось проверить загруженные модели: {r.status_code}")
                return None

            data = r.json()
            models_list = data.get("models", [])

            is_target_loaded = False
            instances_to_unload = []

            for m in models_list:
                m_path = m.get("path") or m.get("id") or m.get("key", "")
                instances = m.get("loaded_instances", [])
                if m_path == full_id and instances:
                    is_target_loaded = True
                    logger.info(f"✅ Модель '{full_id}' уже загружена")
                elif instances:
                    for inst in instances:
                        inst_id = inst.get("instance_id") or inst.get("id")
                        if inst_id:
                            instances_to_unload.append((m_path, inst_id))

            # Выгружаем другие модели
            for m_path, inst_id in instances_to_unload:
                logger.info(f"📤 Выгружаем инстанс '{inst_id}' модели '{m_path}'")
                try:
                    await client.post(
                        f"{API_URL_NATIVE}/models/unload",
                        json={"instance_id": inst_id},
                        headers=headers,
                        timeout=15.0
                    )
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка выгрузки {inst_id}: {e}")

            if is_target_loaded:
                return full_id

            # Загружаем целевую модель
            logger.info(f"📥 Загружаем модель: {full_id}")
            load_payload = {
                "model": full_id,
                "flash_attention": True
            }
            res = await client.post(
                f"{API_URL_NATIVE}/models/load",
                json=load_payload,
                headers=headers,
                timeout=120.0
            )
            if res.status_code == 200:
                logger.info("✅ Команда загрузки принята")

                # ⭐ ИСПРАВЛЕНИЕ: Ждём всего 3 проверки (6 секунд) вместо 5
                # Если модель не появилась — всё равно возвращаем full_id
                logger.info(f"⏳ Ожидаем загрузку модели (до 6 сек)...")
                for i in range(3):
                    await asyncio.sleep(2)
                    try:
                        r = await client.get(f"{API_URL_NATIVE}/models", headers=headers, timeout=10.0)
                        if r.status_code == 200:
                            data = r.json()
                            for m in data.get("models", []):
                                m_path = m.get("path") or m.get("id") or ""
                                if m_path == full_id and m.get("loaded_instances"):
                                    logger.info(f"✅ Модель загружена (попытка {i+1}/3)")
                                    return full_id
                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка проверки загрузки: {e}")
                        continue

                # ⭐ ВАЖНО: Даже если таймаут истёк — возвращаем full_id
                # LM Studio уже получил команду и грузит модель в фоне
                logger.warning(f"⚠️ Таймаут ожидания (6 сек), но продолжаем работу")
                return full_id
            else:
                logger.error(f"❌ Ошибка загрузки модели: {res.status_code} - {res.text}")
                return None
        except Exception as e:
            logger.error(f"❌ Критическая ошибка загрузки модели: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
# --- STREAMING GENERATION (Ключевая фича — время-based обновления) ---
async def stream_generate(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, img_b64: Optional[str] = None) -> str:
    """
    Генерация с поддержкой стриминга через SSE.
    Обновляет сообщение каждые 1.5 секунды (защита от Telegram FloodWait).
    Возвращает финальный ответ для сохранения в историю.
    """
    global last_browser_usage, bot_instance
    sess = sessions[current_chat]

    vision_context = ""

    # Vision processing
    if img_b64 and CFG["ENABLE_VISION"]:
        v_full_id = await ensure_model_loaded("vision")
        if v_full_id:
            async with httpx.AsyncClient(timeout=120) as client:
                payload = {
                    "model": v_full_id,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "Describe the image in Russian."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]}]
                }
                r = await client.post(
                    f"{API_URL_OAI}/chat/completions",
                    headers={"Authorization": f"Bearer {CFG['API_KEY']}"},
                    json=payload,
                    timeout=120.0
                )
                if r.status_code == 200:
                    vision_context = f"[ОПИСАНИЕ ИЗОБРАЖЕНИЯ: {r.json()['choices'][0]['message']['content']}]"
                    logger.info("🖼️ Изображение проанализировано")

    # Загрузка основной модели
    full_id = await ensure_model_loaded(sess.mode)
    if not full_id:
        return "❌ Ошибка: Не удалось загрузить модель в LM Studio. Проверьте логи сервера."

    if sess.mode == "sophia":
        last_browser_usage = time.time()

    messages = sess.get_context(text, vision_context)

    headers = {
        "Authorization": f"Bearer {CFG['API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": full_id,
        "messages": messages,
        "temperature": 0.8,
        "stream": True
    }

    accumulated_content = ""
    reasoning = ""
    start_time = time.time()

    status_msg = await update.message.reply_text("⏳ Генерация...")
    sess.last_message_id = status_msg.message_id
    sess.last_chat_id = update.effective_chat.id

    try:
        async with httpx.AsyncClient(timeout=CFG["TIMEOUT"]) as client:
            async with client.stream("POST", f"{API_URL_OAI}/chat/completions", headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    await status_msg.edit_text(f"❌ API Error {response.status_code}: {error_text[:200]}")
                    return ""

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                accumulated_content += content

                                # ⭐ КЛЮЧЕВОЕ: Обновляем по времени (1.5 сек), а не по количеству чанков
                                current_time = time.time()
                                if current_time - sess.last_update_time >= CFG["STREAM_UPDATE_INTERVAL"]:
                                    display_text = accumulated_content[:4096]
                                    if sess.show_thinking and reasoning:
                                        display_text = f"🧠 *Мысли:*\n_{reasoning[:200]}..._\n{display_text}"
                                    try:
                                        await status_msg.edit_text(display_text)
                                        sess.last_update_time = current_time
                                    except BadRequest as e:
                                        if "message to edit not found" not in str(e).lower():
                                            logger.warning(f"⚠️ Ошибка редактирования: {e}")
                        except json.JSONDecodeError:
                            continue

        # Финальная обработка <think> тегов
        if '<think>' in accumulated_content:
            parts = re.split(r'</?think>', accumulated_content)
            if len(parts) >= 3:
                reasoning = parts[1].strip()
                accumulated_content = (parts[0] + parts[2]).strip()

        # Финальное обновление
        final_text = accumulated_content[:4096]
        if sess.show_thinking and reasoning:
            final_text = f"🧠 *Мысли:*\n_{reasoning[:600]}..._\n{final_text}"

        try:
            await status_msg.edit_text(final_text, parse_mode=ParseMode.MARKDOWN)
        except BadRequest:
            await status_msg.edit_text(final_text, parse_mode=None)

        generation_time = time.time() - start_time
        logger.info(f"✅ Генерация завершена за {generation_time:.1f} сек")

        # Сохраняем в историю
        user_entry = f"{vision_context}\n{text}" if vision_context else text
        sess.add("user", user_entry)
        sess.add("assistant", accumulated_content)
        save_sessions()

        return accumulated_content

    except Exception as e:
        error_msg = f"❌ Ошибка генерации: {str(e)[:200]}"
        logger.error(f"❌ Стриминг упал: {e}")
        try:
            await status_msg.edit_text(error_msg)
        except:
            pass
        return error_msg

# --- NON-STREAMING GENERATION (Fallback) ---
async def generate_response(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, img_b64: Optional[str] = None) -> str:
    """Генерация без стриминга (для отладки или если стриминг отключён)."""
    global last_browser_usage
    sess = sessions[current_chat]

    vision_context = ""
    if img_b64 and CFG["ENABLE_VISION"]:
        v_full_id = await ensure_model_loaded("vision")
        if v_full_id:
            async with httpx.AsyncClient(timeout=120) as client:
                payload = {
                    "model": v_full_id,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "Describe the image in Russian."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]}]
                }
                r = await client.post(f"{API_URL_OAI}/chat/completions", headers={"Authorization": f"Bearer {CFG['API_KEY']}"}, json=payload)
                if r.status_code == 200:
                    vision_context = f"[ОПИСАНИЕ ИЗОБРАЖЕНИЯ: {r.json()['choices'][0]['message']['content']}]"

    full_id = await ensure_model_loaded(sess.mode)
    if not full_id:
        return "❌ Ошибка: Модель не загружена."

    if sess.mode == "sophia":
        last_browser_usage = time.time()

    async with httpx.AsyncClient(timeout=CFG["TIMEOUT"]) as client:
        payload = {
            "model": full_id,
            "messages": sess.get_context(text, vision_context),
            "temperature": 0.8,
            "stream": False
        }
        try:
            r = await client.post(f"{API_URL_OAI}/chat/completions", headers={"Authorization": f"Bearer {CFG['API_KEY']}"}, json=payload)
            if r.status_code != 200:
                return f"❌ API Error {r.status_code}: {r.text}"
            data = r.json()
            content = data['choices'][0]['message'].get('content', '')

            # Обработка <think> тегов
            reasoning = ""
            if '<think>' in content:
                parts = re.split(r'</?think>', content)
                if len(parts) >= 3:
                    reasoning = parts[1].strip()
                    content = (parts[0] + parts[2]).strip()

            final_response = content
            if sess.show_thinking and reasoning:
                final_response = f"🧠 *Мысли:*\n_{reasoning[:600]}..._\n{final_response}"

            user_entry = f"{vision_context}\n{text}" if vision_context else text
            sess.add("user", user_entry)
            sess.add("assistant", content)
            save_sessions()

            return final_response
        except Exception as e:
            return f"❌ Ошибка генерации: {e}"

# --- BACKGROUND TASKS ---
async def browser_killer():
    """Автоматически убивает браузерные процессы после таймаута."""
    global last_browser_usage
    while True:
        await asyncio.sleep(60)
        if last_browser_usage > 0 and (time.time() - last_browser_usage > CFG["BROWSER_TTL"]):
            try:
                import subprocess
                subprocess.run(["pkill", "-f", "chrome"], capture_output=True, timeout=5)
                subprocess.run(["pkill", "-f", "chromium"], capture_output=True, timeout=5)
                logger.info("🧹 Браузерные процессы остановлены (таймаут истёк)")
            except Exception as e:
                logger.debug(f"ℹ️ Ошибка остановки браузера: {e}")

# --- TELEGRAM HANDLERS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != CFG["ADMIN_ID"]:
        await update.message.reply_text("🚫 Доступ запрещён")
        return

    kb = [row[:] for row in KEYBOARD]  # Копия клавиатуры
    sess = sessions.get(current_chat, ChatSession())

    # Обновляем статус Think в клавиатуре
    think_status = "ON" if sess.show_thinking else "OFF"
    for row in kb:
        for i, btn in enumerate(row):
            if "Think:" in btn:
                row[i] = f"🧠 Think: {think_status}"

    await update.message.reply_text(
        "✨ Bridge Online\n"
        "• Стриминг ответов в реальном времени\n"
        "• Автоматическая загрузка/выгрузка моделей\n"
        "• Поддержка изображений (Vision)\n"
        "• Множество сессий и кастомных сценариев RP",
        reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True)
    )

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != CFG["ADMIN_ID"]:
        return

    text = update.message.text.strip()
    global current_chat
    sess = sessions.get(current_chat, ChatSession())
    sessions[current_chat] = sess

    # Режим ввода RP сценария
    if sess.awaiting_rp_input:
        sess.custom_rp = None if text.upper() == "CLEAR" else text
        sess.awaiting_rp_input = False
        save_sessions()
        status = "очищен" if text.upper() == "CLEAR" else "установлен"
        await update.message.reply_text(f"✅ RP сценарий {status}.")
        await show_main_menu(update, sess)
        return

    # ⭐ ДИНАМИЧЕСКИЙ mode_map из config.json
    mode_map = {}
    for row in KEYBOARD:
        for btn_text in row:
            # Пропускаем служебные кнопки
            if btn_text in ["📂 Chats", "🔥 RESET CHAT", "⚙️ Status", "🎭 RP Setup", "📝 Show RP", "❌ Clear RP"]:
                continue
            if btn_text.startswith("🧠 Think:"):
                continue
            # Ищем совпадение по имени модели (убираем эмодзи)
            clean_name = re.sub(r'^[^\w\s]+', '', btn_text).strip().lower()
            for model_key in MODELS.keys():
                if clean_name in model_key.lower() or model_key.lower() in clean_name:
                    mode_map[btn_text] = model_key
                    break

    # ⭐ Переключение режима (ОДИН РАЗ, не дублировать!)
    if text in mode_map:
        sess.mode = mode_map[text]
        save_sessions()
        await update.message.reply_text(f"🔄 Режим изменён на: {sess.mode.upper()}\nМодель будет загружена при следующем запросе.")
        return

    # Остальные команды
    if text == "🔥 RESET CHAT":
        sess.messages = []
        save_sessions()
        await update.message.reply_text("✅ История чата очищена.")
        return

    if text == "📂 Chats":
        btns = [[InlineKeyboardButton(f"{'✅ ' if n==current_chat else ''}{n}", callback_data=f"sw_{n}")] for n in sessions]
        btns.append([InlineKeyboardButton("➕ New", callback_data="new_chat"), InlineKeyboardButton("🗑 Del", callback_data="del_chat")])
        await update.message.reply_text("Выберите сессию:", reply_markup=InlineKeyboardMarkup(btns))
        return

    if text == "⚙️ Status":
        await update.message.reply_text(
            f"📁 Сессия: {current_chat}\n"
            f"🤖 Режим: {sess.mode}\n"
            f"💬 История: {len(sess.messages)} сообщений\n"
            f"💭 Мысли: {'ВКЛ' if sess.show_thinking else 'ВЫКЛ'}\n"
            f"🎭 RP: {'Задан' if sess.custom_rp else 'Нет'}"
        )
        return

    if text.startswith("🧠 Think:"):
        sess.show_thinking = not sess.show_thinking
        save_sessions()
        await show_main_menu(update, sess)
        return

    if text == "🎭 RP Setup":
        sess.awaiting_rp_input = True
        await update.message.reply_text("✏️ Введите сценарий для этого чата.\nОтправьте 'CLEAR' для сброса.")
        return

    if text == "📝 Show RP":
        current_rp = sess.custom_rp or "(не задан)"
        await update.message.reply_text(f"Текущий RP промпт:\n```\n{current_rp}\n```", parse_mode=ParseMode.MARKDOWN)
        return

    if text == "❌ Clear RP":
        sess.custom_rp = None
        save_sessions()
        await update.message.reply_text("✅ RP промпт очищен.")
        return

    # Генерация
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    if CFG["ENABLE_STREAMING"]:
        await stream_generate(update, context, text)
    else:
        resp = await generate_response(update, context, text)
        await safe_send(update, resp)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != CFG["ADMIN_ID"]:
        return

    if not CFG["ENABLE_VISION"]:
        await update.message.reply_text("❌ Vision функция отключена в настройках.")
        return

    photo = update.message.photo[-1]
    caption = update.message.caption or "Проанализируй изображение"

    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.UPLOAD_PHOTO)
    status_msg = await update.message.reply_text("👁 Загружаю изображение...")

    try:
        file = await photo.get_file()
        file_bytes = await file.download_as_bytearray()
        b64 = base64.b64encode(file_bytes).decode('utf-8')
        await status_msg.edit_text("🖼️ Анализирую изображение...")

        if CFG["ENABLE_STREAMING"]:
            await stream_generate(update, context, caption, b64)
        else:
            resp = await generate_response(update, context, caption, b64)
            await status_msg.edit_text(resp)
    except Exception as e:
        await status_msg.edit_text(f"❌ Ошибка обработки изображения: {e}")
        logger.error(f"Ошибка обработки фото: {e}")

async def cb_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    global current_chat

    if query.data.startswith("sw_"):
        target = query.data.replace("sw_", "")
        if target in sessions:
            current_chat = target
            save_sessions()
            await query.edit_message_text(f"✅ Переключено на сессию: {current_chat}")
        else:
            await query.edit_message_text("❌ Сессия не найдена")
    elif query.data == "new_chat":
        new_name = f"Chat_{len(sessions) + 1}"
        sessions[new_name] = ChatSession(new_name)
        current_chat = new_name
        save_sessions()
        await query.edit_message_text(f"✅ Создана сессия: {new_name}")
    elif query.data == "del_chat":
        if current_chat != "Main" and current_chat in sessions:
            del sessions[current_chat]
            current_chat = "Main"
            save_sessions()
            await query.edit_message_text("✅ Сессия удалена. Возврат в Main.")
        else:
            await query.edit_message_text("❌ Нельзя удалить сессию Main")

async def show_main_menu(update: Update, sess: ChatSession):
    kb = [row[:] for row in KEYBOARD]
    think_status = "ON" if sess.show_thinking else "OFF"
    for row in kb:
        for i, btn in enumerate(row):
            if "Think:" in btn:
                row[i] = f"🧠 Think: {think_status}"

    if update.message:
        await update.message.reply_text("Меню обновлено.", reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True))
    elif update.callback_query:
        await update.callback_query.message.reply_text("Меню обновлено.", reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True))

async def safe_send(update: Update, text: str):
    """Безопасная отправка с fallback на plain text."""
    if not text:
        return
    try:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(text, parse_mode=None)

async def post_init(application: Application):
    """Инициализация при запуске бота."""
    global bot_instance
    bot_instance = application.bot
    load_sessions()
    asyncio.create_task(browser_killer())
    logger.info("🚀 Bridge Telegram Bot запущен!")
    logger.info(f"👤 ADMIN_ID: {CFG['ADMIN_ID']}")
    logger.info(f"🌐 LM Studio URL: {CFG['BASE_URL']}")
    logger.info(f"🧠 Доступные режимы: {', '.join(MODELS.keys())}")

# --- MAIN ---
# ✅ FIX ДЛЯ PYTHON 3.14
import sys
if sys.version_info >= (3, 14):
    import asyncio
    # В Python 3.14 нужно явно создавать loop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
if __name__ == "__main__":
    request = HTTPXRequest(
        connection_pool_size=10,
        read_timeout=CFG["TIMEOUT"],
        connect_timeout=30.0,
        write_timeout=30.0
    )
    app = (
        ApplicationBuilder()
        .token(CFG["TOKEN"])
        .request(request)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_msg))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CallbackQueryHandler(cb_handler))

    logger.info("▶️ Запуск бота...")
    try:
        app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("🛑 Бот остановлен пользователем")
    except Exception as e:
        logger.critical(f"💥 Критическая ошибка: {e}")
        raise
