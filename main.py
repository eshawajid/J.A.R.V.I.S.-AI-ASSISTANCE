import os

from ddgs import DDGS
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from elevenlabs.client import ElevenLabs

from memory import init_db, save_message, get_recent_messages


app = FastAPI()


# -------------------------
# Web Search
# -------------------------

def web_search(query):
    try:
        results = DDGS().text(query, max_results=5)

        if not results:
            return "No web results found."

        text = ""

        for result in results:
            text += f"Title: {result.get('title', '')}\n"
            text += f"URL: {result.get('href', '')}\n"
            text += f"Summary: {result.get('body', '')}\n\n"

        return text

    except Exception as e:
        return f"Web search unavailable: {e}"


# -------------------------
# Gemini
# -------------------------

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# -------------------------
# ElevenLabs
# -------------------------

eleven_client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")


# -------------------------
# Initialize Memory
# -------------------------

init_db()


class Message(BaseModel):
    message: str


# -------------------------
# Chat
# -------------------------

@app.post("/chat")
async def chat(data: Message):

    # Save user's message
    save_message("user", data.message)

    # Get recent memory
    recent_messages = get_recent_messages(10)

    conversation = ""

    for role, content in recent_messages:
        conversation += f"{role}: {content}\n"

    # Web search
    search_results = web_search(data.message)

    conversation += f"\nWeb search results:\n{search_results}\n"

    conversation += f"user: {data.message}\nassistant:"

    # Ask Gemini
    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=conversation,
        system_instruction="""You are J.A.R.V.I.S., a personal AI assistant.

Personality:
- Calm, intelligent, professional, and helpful.
- Speak naturally and concisely.
- Address the user as Sir when appropriate, but do not overuse it.
- Do not make jokes or sarcastic comments unless the user clearly asks for humor.
- Do not mention Tony Stark, Iron Man, the Avengers, or fictional JARVIS unless the user specifically asks about them.
- Do not exaggerate simple messages.
- If the user says hello, greet them briefly and naturally.
- Give useful answers rather than theatrical speeches.
- When web search results are provided, use them when they are relevant.
- Do not pretend that you searched the web if the search results are unavailable.
- For simple questions that do not need current information, answer normally.

You are an actual AI assistant helping the user with questions, tasks, information, and their computer project."""
    )

    response = interaction.output_text

    # Save JARVIS response
    save_message("assistant", response)

    return {"response": response}


# -------------------------
# Voice
# -------------------------

@app.post("/speak")
async def speak(data: dict):

    from fastapi.responses import Response, JSONResponse

    try:
        audio = eleven_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=data["text"],
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )

        audio_data = b"".join(audio)

        return Response(
            content=audio_data,
            media_type="audio/mpeg"
        )

    except Exception as e:
        print("ElevenLabs voice error:", e)

        return JSONResponse(
            status_code=503,
            content={
                "error": "ElevenLabs voice is currently unavailable.",
                "message": str(e)
            }
        )


# -------------------------
# Serve JARVIS Website
# -------------------------

app.mount(
    "/",
    StaticFiles(directory="static", html=True),
    name="static"
)
