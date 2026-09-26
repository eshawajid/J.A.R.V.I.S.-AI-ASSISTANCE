import os
from dotenv import load_dotenv

load_dotenv()

from dotenv import load_dotenv

load_dotenv()

from ddgs import DDGS
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
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
# OpenRouter
# -------------------------

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
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
# J.A.R.V.I.S. System Instruction
# -------------------------

SYSTEM_INSTRUCTION = """You are J.A.R.V.I.S., a personal AI assistant.

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

    conversation += f"user: {data.message}"

    # Ask OpenRouter
    response = openrouter_client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION
            },
            {
                "role": "user",
                "content": conversation
            }
        ]
    )

    response_text = response.choices[0].message.content

    # Save JARVIS response
    save_message("assistant", response_text)

    return {"response": response_text}


# -------------------------
# OpenRouter Test
# -------------------------

@app.get("/openrouter-test")
async def openrouter_test():

    try:
        response = openrouter_client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: JARVIS ONLINE"
                }
            ]
        )

        return {
            "success": True,
            "response": response.choices[0].message.content
        }

    except Exception as e:
        return {
            "success": False,
            "error_type": type(e).__name__,
            "error_details": repr(e)
        }


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