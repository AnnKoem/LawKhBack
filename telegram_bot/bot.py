"""
Telegram Bot client for Cambodian Law RAG system.
"""

import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from api_client import query_rag
from config import TELEGRAM_BOT_TOKEN

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to the Cambodian Law Assistant!\n\n"
        "Ask me any question about Cambodian law and I will do my best to help.\n\n"
        "Simply type your question and send it."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "How to use this bot:\n\n"
        "1. Type any question about Cambodian law\n"
        "2. Wait for the response\n\n"
        "Commands:\n"
        "/start - Welcome message\n"
        "/help  - Show this help"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = update.message.text
    user_id = update.effective_user.id

    logger.info("Question from user %s: %s", user_id, question)

    await update.message.chat.send_action("typing")

    try:
        answer = await query_rag(question, user_id)
        await update.message.reply_text(answer)
    except Exception as exc:
        logger.error("Error querying RAG API: %s", exc)
        await update.message.reply_text(
            "Sorry, something went wrong while processing your question. Please try again later."
        )


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
