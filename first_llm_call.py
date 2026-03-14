# first_llm_call.py
# Your first real LLM API call — using the new Google GenAI SDK (google-genai)
# New SDK uses a single client object for all API methods.

import os
from dotenv import load_dotenv
from google import genai





# ============================================================
# SECTION 1: Loading secrets safely
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Make sure your .env file exists and contains GEMINI_API_KEY=your_key"
    )

print(f"✅ API key loaded: {api_key[:8]}...")

# New SDK: one client object handles everything
# All API methods live on this client — models, chats, files, etc.
client = genai.Client(api_key=api_key)


# ============================================================
# SECTION 2: Your first LLM call
# ============================================================

print("\n--- Making first API call ---")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="In exactly 2 sentences, explain what an AI agent is."
)

print(f"Gemini says:\n{response.text}")


# ============================================================
# SECTION 3: Understanding the full response object
# ============================================================

print("\n--- Full response breakdown ---")

response2 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=(
        "List 3 Python frameworks used for building AI agents. "
        "Just the names, one per line, no explanation."
    )
)

print(f"Text:\n{response2.text}")

# Token usage — how many tokens went in and came out
# Tokens ≈ 0.75 words. Important for cost tracking in production.
print(f"Input tokens: {response2.usage_metadata.prompt_token_count}")
print(f"Output tokens: {response2.usage_metadata.candidates_token_count}")


# ============================================================
# SECTION 4: System prompts — giving the model a role
# ============================================================

# A system prompt sets the model's behaviour before the user message.
# This is how you turn a general LLM into a specialised agent.
# Every agent you build will start with a system prompt.

from google.genai import types  # types module holds config objects

print("\n--- With system prompt ---")

code_to_review = """
def get_user(id):
    data = requests.get("https://api.example.com/users/" + id)
    return data.json()
"""

response3 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"Review this code:\n{code_to_review}",
    config=types.GenerateContentConfig(
        # system_instruction = the agent's role, rules, and constraints
        # The model reads this before processing the user message.
        system_instruction=(
            "You are a senior Python developer reviewing code. "
            "You are direct, concise, and always point out exactly one "
            "improvement the developer should make. Nothing more."
        )
    )
)

print(f"Code review:\n{response3.text}")


# ============================================================
# SECTION 5: Multi-turn conversation (chat history)
# ============================================================

# Agents need memory within a session.
# Each message needs context from previous messages.
# The new SDK handles this via client.chats.create()

print("\n--- Multi-turn conversation ---")

# Create a chat session — SDK maintains history automatically
chat = client.chats.create(model="gemini-2.5-flash")

# First turn
response4 = chat.send_message(
    "My name is Anirudh and I'm building an AI agent startup."
)
print(f"Turn 1: {response4.text}")

# Second turn — Gemini remembers the first message automatically
response5 = chat.send_message(
    "What should I focus on learning first?"
)
print(f"Turn 2: {response5.text}")

# Third turn — memory check
response6 = chat.send_message(
    "What did I tell you my name was?"
)
print(f"Turn 3 (memory check): {response6.text}")

# chat.get_history() returns all turns stored in this session
history = chat.get_history()
print(f"\nConversation turns stored: {len(history)}")