# decorators_practice.py
# Decorators: functions that wrap other functions to add behaviour.
# Used everywhere in FastAPI, LangChain, and agent retry logic.


# ============================================================
# SECTION 1: What a decorator actually is
# ============================================================

# Step 1: a normal function
def greet(name: str) -> str:
    return f"Hello, {name}"


# Step 2: a wrapper function that adds behaviour around greet
# It takes the original function as an argument
def add_logging(func):
    # Define a new function that wraps the original
    # *args and **kwargs mean "accept any arguments and pass them through"
    # This makes the wrapper work with ANY function regardless of its parameters
    def wrapper(*args, **kwargs):
        print(f"[LOG] Calling: {func.__name__}")   # runs BEFORE the original
        result = func(*args, **kwargs)              # runs the ORIGINAL function
        print(f"[LOG] Returned: {result}")          # runs AFTER the original
        return result                               # returns the original result
    return wrapper  # returns the new wrapped version


# Step 3: manually applying the wrapper — this is what @ does under the hood
greet_with_logging = add_logging(greet)
greet_with_logging("Anirudh")
# [LOG] Calling: greet
# [LOG] Returned: Hello, Anirudh


# Step 4: the @ syntax is just cleaner shorthand for the same thing
# These two are IDENTICAL:
#
# greet = add_logging(greet)        ← manual way
#
# @add_logging                       ← decorator syntax, does the same thing
# def greet(name: str) -> str:
#     return f"Hello, {name}"

@add_logging
def greet_decorated(name: str) -> str:
    return f"Hello, {name}"

greet_decorated("Modgan")
# [LOG] Calling: greet_decorated
# [LOG] Returned: Hello, Modgan

# ============================================================
# SECTION 2: Timer decorator
# Measures how long any function takes to run.
# ============================================================

print("\n\n\nTime module\n\n")

import time  # built-in Python module — no pip install needed


def timer(func):
    # Exact same structure as add_logging.
    # func = whatever function gets decorated.
    def wrapper(*args, **kwargs):
        start = time.time()              # snapshot BEFORE func runs

        result = func(*args, **kwargs)   # func runs here
                                         # everything inside func happens
                                         # then result gets the return value

        end = time.time()                # snapshot AFTER func runs

        duration = end - start           # how long func took in seconds

        # :.4f = format this float to 4 decimal places
        # Without it you'd see 1.2004785537719727 — too noisy
        # With it you see 1.2005 — clean and readable
        print(f"[TIMER] {func.__name__} took {duration:.4f} seconds")
        print(f"\n\n\n\nresult={result}\ntype={type(result)}\n\n\n")

        return result                    # pass the original return value through
                                         # same reason as Section 1 — don't swallow it
    print(f"\n\n\n\nwrapper{wrapper}\ntype={type(wrapper)}\n\n\n")
    return wrapper  # return the wrapper object, not the result of calling it



# simulate_api_call pretends to be a slow LLM API call.
# Real LLM calls (Gemini, Claude, GPT) take 1-3 seconds.
# time.sleep(1.2) pauses Python for 1.2 seconds — simulating that wait.
@timer
def simulate_api_call(query: str) -> str:
    time.sleep(1.2)                  # freeze execution for 1.2 seconds
    return f"Results for: {query}"


# fast_operation does instant math — no delay.
# This shows the timer works on fast functions too, not just slow ones.
@timer
def fast_operation(x: int, y: int) -> int:
    return x + y


# Call both and watch the timer output
result1 = simulate_api_call("latest AI news")
print(result1)

result2 = fast_operation(10, 20)
print(result2)

# ============================================================
# SECTION 3: Retry decorator
# Automatically retries a function if it raises an exception.
# ============================================================

import time


def retry(max_attempts: int = 3, delay: float = 1.0):
    # LAYER 1: retry() is called first with the parameters.
    # It receives max_attempts and delay.
    # Its only job is to return a decorator.
    # It does NOT receive func here — that happens one layer down.

    def decorator(func):
        # LAYER 2: decorator() receives the actual function.
        # This is the same role add_logging played in Section 1.
        # Its only job is to return wrapper.

        def wrapper(*args, **kwargs):
            # LAYER 3: wrapper() is what actually runs every time
            # the decorated function is called.
            # This is where the retry logic lives.

            last_exception = None
            # We store the last error here.
            # If all attempts fail, we re-raise it at the end
            # so the caller still gets an error — not silent failure.

            for attempt in range(1, max_attempts + 1):
                # range(1, 4) = [1, 2, 3] when max_attempts = 3
                # We start at 1 (not 0) so the print says "Attempt 1/3"
                # instead of "Attempt 0/3" which looks wrong to humans.

                try:
                    result = func(*args, **kwargs)
                    # Try running the function.
                    # If it succeeds, return immediately.
                    # The loop stops. No more retries needed.
                    return result

                except Exception as e:
                    # If the function raised ANY exception, we land here.
                    last_exception = e  # store it in case this is the last attempt
                    print(f"[RETRY] Attempt {attempt}/{max_attempts} failed: {e}")

                    if attempt < max_attempts:
                        # Don't sleep after the LAST attempt — pointless waiting
                        print(f"[RETRY] Waiting {delay}s before retry...")
                        time.sleep(delay)

            # If we reach here, all attempts failed.
            # Raise the last exception so the caller knows something went wrong.
            print(f"[RETRY] All {max_attempts} attempts failed.")
            raise last_exception

        return wrapper   # decorator returns wrapper
    return decorator     # retry returns decorator


# ---- Simulating a flaky API ----
# We need a way to make a function fail twice then succeed.
# attempt_counter tracks how many times the function has been called.
# It lives OUTSIDE the function so it persists between calls.
attempt_counter = 0


@retry(max_attempts=3, delay=0.5)
def flaky_api_call(query: str) -> str:
    global attempt_counter
    # global tells Python: "attempt_counter refers to the variable
    # defined OUTSIDE this function, not a new local one."
    # Without global, Python would create a NEW local attempt_counter
    # that resets to 0 every call — the counter would never increment.

    attempt_counter += 1

    if attempt_counter < 3:
        # Fail on attempts 1 and 2
        raise ConnectionError(f"API timeout on attempt {attempt_counter}")

    # Succeed on attempt 3
    return f"Success: {query}"


result = flaky_api_call("search query")
print(f"Final result: {result}")

# ============================================================
# SECTION 4: Stacking decorators
# ============================================================

# Reset counter for a clean demo
attempt_counter = 0


@timer                              # OUTER — runs first, wraps everything
@retry(max_attempts=3, delay=0.3)   # INNER — runs second, wraps search_web
def search_web(query: str) -> str:
    global attempt_counter
    attempt_counter += 1

    if attempt_counter < 2:
        # Fail once, then succeed
        raise TimeoutError("Search API timed out")

    time.sleep(0.5)   # simulate real network delay after success
    return f"Search results for: {query}"


result = search_web("agentic AI frameworks")
print(f"Result: {result}")

# ============================================================
# SECTION 5: Preview — building a simplified @tool decorator
# LangChain isn't installed yet. We're building the concept manually.
# In week 5 you'll import the real one — it works identically.
# ============================================================

def tool(func):
    # This decorator doesn't wrap the function with new behaviour.
    # Instead it TAGS the function with metadata so an agent framework
    # can discover it, read its capabilities, and decide when to use it.

    func.is_tool = True
    # Adds a new attribute to the function object.
    # func.is_tool = True means "this function is registered as an agent tool"
    # An agent framework can check: if func.is_tool → include in available tools

    func.tool_name = func.__name__
    # Stores the function's name as an attribute.
    # The agent uses this as the tool's identifier.
    # When the LLM says "use search_database", the framework matches
    # this string to find the right function to call.

    func.description = func.__doc__
    # __doc__ is the function's docstring — the string written right
    # after the def line in triple quotes.
    # This is CRITICAL. The LLM reads this description to decide
    # whether to use this tool for a given task.
    # Bad docstring = LLM uses the tool at wrong times or never.
    # Good docstring = LLM knows exactly when and how to use it.

    func.input_schema = {
        # Reads type hints and builds a schema the LLM understands.
        # In real LangChain this is a full Pydantic model (week 3 connects here).
        # We're simplifying — just storing the hint annotations directly.
        param: str(hint)
        for param, hint in func.__annotations__.items()
        if param != "return"  # exclude the return type, only want input params
    }

    return func
    # Unlike Sections 1-3, we return the ORIGINAL func unchanged.
    # We're not wrapping it — we're just attaching metadata to it.
    # The function behaves exactly as before when called.
    # The difference is it now carries extra attributes the framework reads.


# Three different tools — notice how the docstrings are the key differentiator.
# The LLM reads ONLY the docstring to decide which tool to use.

@tool
def search_database(query: str) -> str:
    """Search the product database for items matching the query."""
    return f"Found 3 results for: {query}"


@tool
def send_email(recipient: str, subject: str, body: str) -> bool:
    """Send an email to a recipient with a given subject and body."""
    print(f"Email sent to {recipient}: {subject}")
    return True


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"Weather in {city}: 28°C, partly cloudy"


# How an agent framework discovers available tools:
all_tools = [search_database, send_email, get_weather]

print("=== Available Tools ===")
for t in all_tools:
    print(f"\nName:        {t.tool_name}")
    print(f"Description: {t.description}")
    print(f"Inputs:      {t.input_schema}")
    print(f"Is tool:     {t.is_tool}")

# The agent framework passes this list to the LLM.
# The LLM reads names and descriptions, picks the right tool,
# and calls it with the right parameters.

print("\n=== Calling Tools Normally ===")
# Tools still work as regular functions — the decorator didn't change that
print(search_database("wireless headphones"))
print(get_weather("Chennai"))