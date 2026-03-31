# multi_turn.py
# Multi-turn conversations and context management.
# Short-term memory for agents — how to maintain and trim history.

import json
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen2.5:7b"

# The messages format — a list of dicts, each with role and content.
# This is the universal format across ALL LLM APIs.
# role = "system"    → sets agent behaviour (sent once at the start)
# role = "user"      → human message
# role = "assistant" → model's previous response
#
# The model sees the ENTIRE list on every call.
# That's how it "remembers" — you're sending the full history each time.
# There is no server-side memory. You own and manage the history.

Messages = list[dict[str, str]]  # type alias for readability


# ============================================================
# SECTION 1: Basic multi-turn conversation
# ============================================================

print("=" * 50)
print("SECTION 1: Basic multi-turn")
print("=" * 50)


def chat(
    messages: Messages,
    system: str,
    user_input: str
) -> tuple[str, Messages]:
    """
    Send one turn of a conversation.

    Takes the full history, adds the new user message,
    calls the model, appends the response, returns both
    the response text and the updated history.

    Returns:
        tuple of (response_text, updated_messages)
    """
    # Add new user message to history
    messages = messages + [{"role": "user", "content": user_input}]
    # Note: we use + not .append() to avoid mutating the original list
    # This makes the function pure — no side effects on the input

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}] + messages,
        temperature=0.7,
        max_tokens=3000
    )

    assistant_reply = response.choices[0].message.content.strip()

    # Add assistant response to history
    messages = messages + [{"role": "assistant", "content": assistant_reply}]

    return assistant_reply, messages


# Simulate a multi-turn conversation
system = """You are a concise technical mentor helping a developer learn
agentic AI. Keep responses under 3 sentences. Be direct."""

history: Messages = []

turns = [
    "What is the most important concept in agentic AI?",
    "Can you give me a concrete example of that?",
    "How long would it take to learn this?",
    "What should I build first to practice?",
]

print(f"\nSystem: {system}\n")
for user_msg in turns:
    print(f"User: {user_msg}")
    reply, history = chat(history, system, user_msg)
    print(f"Assistant: {reply}\n")

print(f"Total messages in history: {len(history)}")
print(f"History structure:")
for msg in history:
    preview = msg['content'][:50].replace('\n', ' ')
    print(f"  [{msg['role']}]: {preview}...")


# ============================================================
# SECTION 2: Token counting and context limits
# ============================================================

print("\n" + "=" * 50)
print("SECTION 2: Token counting")
print("=" * 50)

def count_tokens_approximate(messages: Messages, system: str) -> int:
    """
    Approximate token count for a conversation.
    Real count requires a tokenizer — this is close enough for management.
    Rule of thumb: 1 token ≈ 4 characters
    """
    total_chars = len(system)
    for msg in messages:
        total_chars += len(msg["content"])
    return total_chars // 4  # approximate tokens


tokens = count_tokens_approximate(history, system)
print(f"\nApproximate tokens in current conversation: {tokens}")
print(f"qwen2.5:7b context window: ~32,000 tokens")
print(f"Remaining capacity: ~{32000 - tokens} tokens")

# At this rate, how many more turns before hitting the limit?
avg_tokens_per_turn = tokens / len(history) if history else 0
turns_remaining = (32000 - tokens) / avg_tokens_per_turn if avg_tokens_per_turn else 0
print(f"Average tokens per turn: {avg_tokens_per_turn:.0f}")
print(f"Estimated turns before limit: {turns_remaining:.0f}")


# ============================================================
# SECTION 3: History trimming strategies
# ============================================================

print("\n" + "=" * 50)
print("SECTION 3: History trimming")
print("=" * 50)

# When history gets too long you have three strategies:
# 1. Sliding window — keep only the last N messages
# 2. Summarise — compress old turns into a summary
# 3. Selective — keep first turn (context) + last N turns (recent)

def trim_sliding_window(messages: Messages, max_messages: int) -> Messages:
    """
    Keep only the last max_messages messages.
    Simple but loses early context.
    Must keep pairs (user+assistant) so use even numbers.
    """
    if len(messages) <= max_messages:
        return messages
    # Always trim to even number to keep user/assistant pairs intact
    trimmed = messages[-max_messages:]
    print(f"  Trimmed from {len(messages)} to {len(trimmed)} messages")
    return trimmed


def trim_keep_first_and_last(
    messages: Messages,
    keep_first: int = 2,
    keep_last: int = 6
) -> Messages:
    """
    Keep the first N and last N messages.
    Preserves initial context (what the conversation is about)
    while keeping recent context (what was just said).
    """
    if len(messages) <= keep_first + keep_last:
        return messages

    first = messages[:keep_first]
    last = messages[-keep_last:]

    # Add a marker so the model knows history was trimmed
    marker = {
        "role": "system",
        "content": f"[{len(messages) - keep_first - keep_last} earlier messages omitted]"
    }

    trimmed = first + [marker] + last
    print(f"  Kept first {keep_first} + last {keep_last} of {len(messages)} messages")
    return trimmed


# Demonstrate both strategies
print(f"\nOriginal history length: {len(history)} messages")

window_trimmed = trim_sliding_window(history, max_messages=4)
print(f"After sliding window (max=4): {len(window_trimmed)} messages")

selective_trimmed = trim_keep_first_and_last(history, keep_first=2, keep_last=4)
print(f"After selective trim: {len(selective_trimmed)} messages")


# ============================================================
# SECTION 4: Conversation class — managing state cleanly
# ============================================================

print("\n" + "=" * 50)
print("SECTION 4: Conversation class")
print("=" * 50)

class Conversation:
    """
    Manages a multi-turn conversation with automatic history trimming.
    This is the pattern agent frameworks use internally —
    LangChain's ConversationChain, CrewAI's agent memory,
    all wrap this same concept.
    """

    def __init__(
        self,
        system: str,
        max_history: int = 20,
        model: str = MODEL
    ):
        self.system = system
        self.max_history = max_history
        self.model = model
        self.history: Messages = []
        self.turn_count = 0

    def send(self, user_input: str) -> str:
        """Send a message and get a response."""
        self.turn_count += 1

        # Add user message
        self.history.append({"role": "user", "content": user_input})

        # Trim if needed (before the call — don't waste tokens)
        if len(self.history) > self.max_history:
            self.history = trim_sliding_window(
                self.history,
                self.max_history
            )

        # Make the call
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system}] + self.history,
            temperature=0.7,
            max_tokens=3000
        )

        reply = response.choices[0].message.content.strip()

        # Add assistant response to history
        self.history.append({"role": "assistant", "content": reply})

        return reply

    def get_summary(self) -> dict:
        """Return conversation stats."""
        return {
            "turns": self.turn_count,
            "messages_in_history": len(self.history),
            "approx_tokens": count_tokens_approximate(self.history, self.system)
        }

    def reset(self) -> None:
        """Start fresh while keeping the system prompt."""
        self.history = []
        self.turn_count = 0


# Test the Conversation class
convo = Conversation(
    system="You are a Socratic tutor. Ask one question back after every answer.",
    max_history=10
)

print("\nConversation with Socratic tutor:")
exchanges = [
    "I want to learn about neural networks",
    "I think they're inspired by the brain",
    "Layers of connected nodes that process data",
]

for msg in exchanges:
    print(f"\nUser: {msg}")
    reply = convo.send(msg)
    print(f"Tutor: {reply}")

print(f"\nConversation stats: {convo.get_summary()}")