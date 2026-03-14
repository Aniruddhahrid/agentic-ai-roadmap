# type_hints_practice.py
# Today we learn to read and write type hints.
# These appear everywhere in agent framework code.


# ============================================================
# SECTION 1: Basic type hints
# ============================================================

# Without type hints — valid Python, but tells you nothing
def add(a, b):
    return a + b


# With type hints — instantly clear what goes in and what comes out
# int = integer number
# -> int = this function returns an integer
def add_typed(a: int, b: int) -> int:
    return a + b


# str = string (text)
def greet(name: str) -> str:
    return f"Hello, {name}"


# bool = True or False
def is_adult(age: int) -> bool:
    return age >= 18


# float = decimal number
def calculate_score(raw: int, multiplier: float) -> float:
    return raw * multiplier


# None = this function returns nothing (just does something and exits)
def log_message(message: str) -> None:
    print(f"[LOG]: {message}")


# Test all of them
print(add_typed(2, 3))           # 5
print(greet("Anirudh"))          # Hello, Anirudh
print(is_adult(20))              # True
print(calculate_score(85, 1.5))  # 127.5
log_message("Setup complete")    # [LOG]: Setup complete

# ============================================================
# SECTION 2: Container types
# ============================================================

# list[str] = a list where every item is a string
# Without [str] you'd just write "list" — vague and unhelpful
# list[str] tells you and VS Code exactly what's inside
def join_names(names: list[str]) -> str:
    return ", ".join(names)


# dict[str, int] = dictionary where keys are strings, values are integers
# Example: {"alice": 95, "bob": 87}
def get_top_scorer(scores: dict[str, int]) -> str:
    return max(scores, key=scores.get)


# list[dict[str, str]] = a list of dictionaries where keys and values are both strings
# THIS IS EXACTLY the format LangChain uses for conversation history:
# [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
# You'll write this type hint dozens of times across the 12 weeks
def format_history(messages: list[dict[str, str]]) -> str:
    result = ""
    for msg in messages:
        result += f"{msg['role']}: {msg['content']}\n"
    return result


# Test them
print(join_names(["Alice", "Bob", "Charlie"]))
# Alice, Bob, Charlie

scores = {"alice": 95, "bob": 87, "charlie": 92}
print(get_top_scorer(scores))
# alice

history = [
    {"role": "user", "content": "What is an agent?"},
    {"role": "assistant", "content": "An agent is an AI that takes actions."}
]
print(format_history(history))
# user: What is an agent?
# assistant: An agent is an AI that takes actions.

# ============================================================
# SECTION 3: Optional — when a value might be None
# ============================================================

# Import Optional from Python's built-in typing module
from typing import Optional

# Optional[str] means: this returns either a str OR None
# Without Optional the return type hint would be a lie
def find_name(names: list[str], search: str) -> Optional[str]:
    for name in names:
        if name.lower() == search.lower():
            return name
    return None  # explicitly return None when not found


# When a function returns Optional, always check for None before using the result.
# This is called a "None check" — you'll write this pattern constantly.
result = find_name(["Alice", "Bob", "Charlie"], "bob")
if result is not None:
    print(f"Found: {result}")   # Found: Bob
else:
    print("Not found")

result2 = find_name(["Alice", "Bob", "Charlie"], "dave")
if result2 is not None:
    print(f"Found: {result2}")
else:
    print("Not found")          # Not found


# Optional with a default parameter value
# = None gives it a default so callers don't have to pass it every time
def greet_user(name: str, title: Optional[str] = None) -> str:
    if title:
        return f"Hello, {title} {name}"
    return f"Hello, {name}"


print(greet_user("Anirudh"))          # Hello, Anirudh
print(greet_user("Anirudh", "Mr."))   # Hello, Mr. Anirudh

# ============================================================
# SECTION 4: Realistic agent-style function signature
# ============================================================

# This is the shape of a real LLM call wrapper.
# Read just the type hints — you know exactly what this function
# needs and returns without reading a single line of the body.

def run_agent_task(
    task: str,                            # the instruction to the agent
    history: list[dict[str, str]],        # previous conversation turns
    max_tokens: int = 1000,               # how long the response can be
    temperature: float = 0.7,             # randomness: 0=deterministic, 1=creative
    system_prompt: Optional[str] = None   # optional override for agent behaviour
) -> dict[str, str]:                      # returns one message dict: role + content

    # Mock implementation — no real LLM yet, that's week 3.
    # The lesson today is reading and writing the signature, not the body.
    return {
        "role": "assistant",
        "content": f"Processed task: {task}"
    }


# Calling it — the parameter names + types make this self-documenting
response = run_agent_task(
    task="Summarise the latest AI news",
    history=[{"role": "user", "content": "Hello"}],
    max_tokens=500,
    temperature=0.3
)
print(response)
# {'role': 'assistant', 'content': 'Processed task: Summarise the latest AI news'}

print("\n✅ Day 1 complete. Type hints practiced")