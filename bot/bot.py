import telebot
import requests
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
import re

# НАСТРОЙКИ
TOKEN = ""
API_URL = "http://127.0.0.1:8000/predict"

bot = telebot.TeleBot(TOKEN)


# КЛАВИАТУРА
def main_keyboard():
    markup = ReplyKeyboardMarkup(resize_keyboard=True)

    markup.add(
        KeyboardButton("🧠 Проверить новость"),
        KeyboardButton("ℹ️ Инструкция")
    )

    return markup


# INLINE КНОПКИ
def result_keyboard():
    markup = InlineKeyboardMarkup()

    markup.add(
        InlineKeyboardButton("🔁 Проверить ещё", callback_data="again")
    )

    return markup


# ВАЛИДАЦИЯ ТЕКСТА
def validate_text(text: str):
    if not text:
        return False, "❌ Пустой текст"

    text = text.strip()

    # удаляем лишние пробелы
    text = re.sub(r"\s+", " ", text)

    words = text.split()

    # минимум слов
    if len(words) < 5:
        return False, "⚠️ Слишком короткий текст. Нужно минимум 5 слов."

    # проверка на мусор
    alpha_words = [w for w in words if re.search(r"[a-zA-Zа-яА-Я]", w)]
    if len(alpha_words) < 3:
        return False, "⚠️ Текст не похож на новость."

    return True, text


# START
@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(
        message.chat.id,
        "👋 Привет! Я определяю фейковые новости.\n\n"
        "Отправь текст новости — и я проведу анализ.",
        reply_markup=main_keyboard()
    )


# ИНСТРУКЦИЯ
@bot.message_handler(func=lambda m: m.text == "ℹ️ Инструкция")
def help(message):
    bot.send_message(
        message.chat.id,
        "📌 Как пользоваться:\n\n"
        "• Отправь новость (минимум 5 слов)\n"
        "• Получи результат\n"
        "• Используй кнопку «Проверить ещё»\n\n"
        "⚠️ Работает только с текстом новостей."
    )


#  МЕНЮ 
@bot.message_handler(func=lambda m: m.text == "🧠 Проверить новость")
def ask_news(message):
    bot.send_message(
        message.chat.id,
        "📨 Отправь текст новости одним сообщением."
    )


# ОСНОВНАЯ ЛОГИКА
@bot.message_handler(func=lambda message: True)
def handle_message(message):

    valid, processed = validate_text(message.text)

    if not valid:
        bot.send_message(message.chat.id, processed)
        return

    text = processed

    try:
        response = requests.post(
            API_URL,
            json={"text": text},
            timeout=10
        )

        data = response.json()

        label = data["label"]
        fake_p = data["fake_probability"]
        real_p = data["real_probability"]

        emoji = "🚨" if label == "FAKE" else "✅"

        answer = (
            f"{emoji} <b>Результат: {label}</b>\n\n"
            f"📉 Fake: <b>{fake_p:.3f}</b>\n"
            f"📈 Real: <b>{real_p:.3f}</b>"
        )

        bot.send_message(
            message.chat.id,
            answer,
            parse_mode="HTML",
            reply_markup=result_keyboard()
        )

    except Exception as e:
        bot.send_message(
            message.chat.id,
            f"⚠️ Ошибка API:\n{str(e)}"
        )


# CALLBACK
@bot.callback_query_handler(func=lambda call: True)
def callback(call):

    if call.data == "again":
        bot.send_message(
            call.message.chat.id,
            "📨 Отправь новую новость для проверки."
        )


# RUN 
if __name__ == "__main__":
    print("Bot started...")
    bot.infinity_polling()