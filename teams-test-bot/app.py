# app.py
# Entry point for the bot (Flask + Bot Framework SDK)
import os
import asyncio
from flask import Flask, request, Response
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity
from bot import AiBot
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Adapter settings (empty AppId and Password are fine for local testing)
SETTINGS = BotFrameworkAdapterSettings(
    os.getenv("MicrosoftAppId"), 
    os.getenv("MicrosoftAppPassword")
)
ADAPTER = BotFrameworkAdapter(SETTINGS)
BOT = AiBot()

@app.route("/api/messages", methods=["POST"])
def messages():
    if "application/json" not in request.headers.get("Content-Type", ""):
        return Response(status=415)

    body = request.json
    print("Incoming activity:", body)  # Debug print

    activity = Activity().deserialize(body)

    async def aux_func(turn_context: TurnContext):
        await BOT.on_turn(turn_context)

    try:
        # Run the async adapter call
        asyncio.run(ADAPTER.process_activity(activity, "", aux_func))
        return Response(status=201)
    except Exception as e:
        print("❌ Error processing activity:", e)
        traceback.print_exc()
        return Response(str(e), status=500)

if __name__ == "__main__":
    print("Starting Flask bot on http://localhost:3978 ...")
    app.run(host="0.0.0.0", port=3978)
