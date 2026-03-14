# async_practice.py
# Async/Await: how Python does multiple things without blocking.
# Every agent framework is built on this. Learn it once, use it everywhere.

import asyncio  # built-in Python module that runs async code
                # think of it as the "engine" that manages the switching


# ============================================================
# SECTION 1: sync vs async — the core difference
# ============================================================

import time


# SYNCHRONOUS version — blocks while waiting
# Nothing else can happen while this function runs
def fetch_sync(name: str, wait: float) -> str:
    print(f"[SYNC] Starting {name}")
    time.sleep(wait)              # blocks everything for `wait` seconds
                                  # Python sits completely idle here
    print(f"[SYNC] Done {name}")
    return f"Result from {name}"


# ASYNCHRONOUS version — releases control while waiting
# async def = "this function can be paused and resumed"
async def fetch_async(name: str, wait: float) -> str:
    print(f"[ASYNC] Starting {name}")
    await asyncio.sleep(wait)     # "pause THIS function for `wait` seconds"
                                  # BUT release control so other functions can run
                                  # asyncio.sleep is the async version of time.sleep
    print(f"[ASYNC] Done {name}")
    return f"Result from {name}"


# --- Run sync version ---
print("=== SYNCHRONOUS ===")
start = time.time()
fetch_sync("Task A", 1.0)
fetch_sync("Task B", 1.0)
fetch_sync("Task C", 1.0)
print(f"Sync total: {time.time() - start:.2f}s")
# Each task waits for the previous. Total: ~3 seconds.


# --- Run async version ---
# async functions can't be called directly like normal functions.
# You need asyncio.run() to start the async engine and run them.
async def main_basic():
    start = time.time()
    await fetch_async("Task A", 1.0)
    await fetch_async("Task B", 1.0)
    await fetch_async("Task C", 1.0)
    print(f"Async sequential total: {time.time() - start:.2f}s")
    # Still ~3 seconds — we're awaiting one at a time, not concurrently yet
    # Concurrency comes in Section 2

print("\n=== ASYNC SEQUENTIAL ===")
asyncio.run(main_basic())
# asyncio.run() = "start the async engine and run this coroutine"
# You call this exactly once, at the top level of your program

# ============================================================
# SECTION 2: asyncio.gather — running things concurrently
# This is where async actually saves time
# ============================================================

async def main_concurrent():
    start = time.time()

    # asyncio.gather() takes multiple coroutines and runs them CONCURRENTLY.
    # It starts all of them, then manages the switching automatically.
    # When Task A is waiting, Task B runs. When B is waiting, C runs. Etc.
    results = await asyncio.gather(
        fetch_async("Task A", 1.0),
        fetch_async("Task B", 1.0),
        fetch_async("Task C", 1.0),
    )
    # All three start almost simultaneously.
    # Each waits 1 second.
    # Total time: ~1 second, not 3.

    print(f"Async concurrent total: {time.time() - start:.2f}s")
    print(f"Results: {results}")
    # results is a list containing the return value of each coroutine
    # in the same order they were passed to gather()


print("\n=== ASYNC CONCURRENT ===")
asyncio.run(main_concurrent())

# ============================================================
# SECTION 3: Real async HTTP calls
# This is exactly how agent tools call external APIs
# ============================================================

import httpx  # async-compatible HTTP library


async def fetch_url(url: str) -> str:
    # async with = async version of the regular "with" context manager
    # httpx.AsyncClient() = an HTTP client that supports async
    # It opens a connection, we use it, it closes automatically
    async with httpx.AsyncClient() as client:
        print(f"Fetching: {url}")
        response = await client.get(url)
        # await here = "send the HTTP request and wait for response"
        # BUT release control while waiting so other coroutines can run

        # response.status_code: 200 = success, 404 = not found, 500 = server error
        return f"Status {response.status_code} from {url}"


async def main_http():
    start = time.time()

    # Fetch three URLs concurrently — all three requests go out simultaneously
    # In an agent, these would be three different tool calls or API requests
    results = await asyncio.gather(
        fetch_url("https://httpbin.org/delay/1"),  # intentionally slow endpoint
        fetch_url("https://httpbin.org/get"),
        fetch_url("https://httpbin.org/status/200"),
    )

    for result in results:
        print(result)

    print(f"Total time: {time.time() - start:.2f}s")
    # Should be ~1-2s despite fetching 3 URLs
    # because all three requests go out concurrently


print("\n=== REAL HTTP CALLS ===")
asyncio.run(main_http())

# ============================================================
# SECTION 4: The agent tool-calling pattern
# This is what happens inside LangGraph when an agent
# decides to call multiple tools at once
# ============================================================

# Simulated agent tools — each takes time like a real API call would
async def search_web(query: str) -> str:
    await asyncio.sleep(0.8)  # simulates web search latency
    return f"Web results for: {query}"

async def search_database(query: str) -> str:
    await asyncio.sleep(0.5)  # simulates DB query latency
    return f"DB results for: {query}"

async def get_user_context(user_id: str) -> str:
    await asyncio.sleep(0.3)  # simulates profile lookup latency
    return f"Context for user: {user_id}"


async def run_agent_tools(task: str, user_id: str) -> dict:
    print(f"Agent received task: {task}")
    print("Calling all tools concurrently...")

    start = time.time()

    # All three tools fire simultaneously
    # Agent doesn't wait for web search before starting DB search
    web_result, db_result, user_context = await asyncio.gather(
        search_web(task),
        search_database(task),
        get_user_context(user_id),
    )
    # web_result, db_result, user_context are unpacked from the results list
    # in the same order as the gather() arguments

    elapsed = time.time() - start
    print(f"All tools completed in {elapsed:.2f}s")
    # ~0.8s (slowest tool) instead of 1.6s (sum of all tools)

    return {
        "task": task,
        "web": web_result,
        "database": db_result,
        "user": user_context,
        "time_taken": f"{elapsed:.2f}s"
    }


print("\n=== AGENT TOOL CALLING PATTERN ===")
result = asyncio.run(run_agent_tools("latest AI frameworks", "user_123"))
for key, value in result.items():
    print(f"{key}: {value}")