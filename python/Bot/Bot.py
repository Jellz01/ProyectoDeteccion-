import os
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# CONFIGURACIÓN BÁSICA
TOKEN = "8509450066:AAHYzJcxdRjIS7XtR58I-f5Y_xKz6MXmIO8"   
IMAGES_DIR = "/home/jellz/Documents/VisionPorComputador/ProyectoDeteccion/imagenes/true_pedestrians"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # MENSAJE DE PRUEBA PARA CONFIRMAR QUE ES EL NUEVO CÓDIGO
    await update.message.reply_text("✨ ¡SISTEMA ABIERTO! No hay restricciones de ID.\nUsa /stream para iniciar.")

async def stream(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📡 Buscando imágenes en la carpeta de la GPU...")
    sent_files = set()
    while True:
        if os.path.exists(IMAGES_DIR):
            files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
            for f in sorted(files):
                path = os.path.join(IMAGES_DIR, f)
                if path not in sent_files:
                    with open(path, "rb") as img:
                        await context.bot.send_photo(
                            chat_id=update.effective_chat.id, 
                            photo=img, 
                            caption=f"✅ Detección PyTorch: {f}"
                        )
                    sent_files.add(path)
                    await asyncio.sleep(0.5)
        await asyncio.sleep(1)

if __name__ == "__main__":
    print("🚀 ARRANCANDO BOT TOTALMENTE PÚBLICO...")
    # drop_pending_updates=True limpia los mensajes de "Acceso denegado" que quedaron en la cola
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stream", stream))
    app.run_polling(drop_pending_updates=True)