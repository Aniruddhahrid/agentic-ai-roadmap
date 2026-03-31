# streaming.py
# Streaming responses — real-time output from LLMs.
# Makes agents feel responsive rather than frozen.

import sys
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen2.5:7b"


# ============================================================
# SECTION 1: Basic streaming
# ============================================================

print("=" * 50)
print("SECTION 1: Basic streaming")
print("=" * 50)

print("\nNon-streaming (waits for full response):")
start = time.time()
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "List 5 benefits of async programming."}],
    max_tokens=2000,
    stream=False   # default — returns all at once
)
print(f"Time to first token: {time.time() - start:.2f}s")
print(response.choices[0].message.content)


print("\n\nStreaming (words appear as generated):")
start = time.time()
first_token_time = None

stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "List 5 benefits of async programming."}],
    max_tokens=2000,
    stream=True    # enables streaming — returns an iterator
)

# stream is now an iterator of chunks
# each chunk contains a small piece of the response
full_response = ""
for chunk in stream:
    # chunk.choices[0].delta.content = the new text in this chunk
    # delta = "the change" — just the new tokens, not the full text
    content = chunk.choices[0].delta.content

    if content is not None:
        if first_token_time is None:
            first_token_time = time.time()
            print(f"Time to first token: {first_token_time - start:.2f}s")

        # Print without newline, flush immediately so it appears in real time
        # sys.stdout.flush() forces the output buffer to display now
        print(content, end="", flush=True)
        full_response += content

print(f"\n\nFull response length: {len(full_response)} characters")


# ============================================================
# SECTION 2: Streaming with a callback
# ============================================================

print("\n" + "=" * 50)
print("SECTION 2: Streaming with callback")
print("=" * 50)

# In a real app (FastAPI, web server) you don't print to terminal.
# You pass each chunk to a callback function that sends it
# to the frontend via WebSocket or Server-Sent Events.

def stream_with_callback(
    prompt: str,
    on_token: callable,
    on_complete: callable
) -> str:
    """
    Stream a response and call callbacks for each token.

    Args:
        prompt: the user message
        on_token: called for each new token (chunk of text)
        on_complete: called when streaming finishes with full text
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        stream=True
    )

    full_text = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            on_token(content)      # call the callback with new text
            full_text += content

    on_complete(full_text)         # call when done with full response
    return full_text


# Example callbacks
token_count = 0

def handle_token(text: str) -> None:
    """Simulate sending token to a frontend."""
    global token_count
    token_count += 1
    print(text, end="", flush=True)

def handle_complete(full_text: str) -> None:
    """Called when streaming finishes."""
    print(f"\n\n[Stream complete — {token_count} chunks received]")
    print(f"[Total length: {len(full_text)} characters]")


print("\nStreaming with callbacks:")
stream_with_callback(
    prompt="Explain what a Pydantic model is in simple terms.",
    on_token=handle_token,
    on_complete=handle_complete
)


# ============================================================
# SECTION 3: Early termination
# ============================================================

print("\n" + "=" * 50)
print("SECTION 3: Early termination")
print("=" * 50)

# Sometimes you want to stop streaming when you've seen enough.
# Example: streaming a JSON object — stop after the closing brace.

def stream_until(prompt: str, stop_marker: str) -> str:
    """
    Stream response but stop when stop_marker appears.
    Useful for extracting just the JSON part of a response.
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        stream=True
    )

    collected = ""
    stopped_early = False

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            collected += content
            print(content, end="", flush=True)

            # Check if stop marker appeared in collected text
            if stop_marker in collected:
                # Find where the marker ends and truncate
                stop_index = collected.index(stop_marker) + len(stop_marker)
                collected = collected[:stop_index]
                stopped_early = True
                break  # exit the streaming loop early

    print(f"\n\n[Stopped early: {stopped_early}]")
    return collected


print("\nStreaming JSON with early stop at '}':")
result = stream_until(
    prompt='Return a JSON object with "name" and "score" fields only. Then write a long explanation.',
    stop_marker="}"
)
print(f"Captured: {result}")


# ============================================================
# SECTION 4: Streaming in a multi-turn conversation
# ============================================================

print("\n" + "=" * 50)
print("SECTION 4: Streaming multi-turn")
print("=" * 50)

def stream_chat(messages: list, system: str, user_input: str) -> tuple[str, list]:
    """
    Multi-turn chat with streaming output.
    Combines Day 19 conversation management with streaming.
    """
    messages = messages + [{"role": "user", "content": user_input}]

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}] + messages,
        temperature=0.7,
        max_tokens=200,
        stream=True
    )

    print(f"User: {user_input}")
    print("Assistant: ", end="", flush=True)

    full_reply = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_reply += content

    print()  # newline after streaming completes

    messages = messages + [{"role": "assistant", "content": full_reply}]
    return full_reply, messages


system = "You are a concise Python tutor. Keep answers under 2 sentences."
history = []

questions = [
    "What is a decorator?",
    "Can you give a one-line example?",
]

print("\nStreaming multi-turn conversation:")
for question in questions:
    _, history = stream_chat(history, system, question)
    print()