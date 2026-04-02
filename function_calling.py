# function_calling.py
# Function calling — giving LLMs the ability to call your Python functions.
# This is the mechanism behind every agent tool in LangChain, CrewAI, LangGraph.

import json
from typing import Any
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen2.5:7b"


# ============================================================
# SECTION 1: Defining tools
# ============================================================

# print("=" * 50)
# print("SECTION 1: Defining tools")
# print("=" * 50)

# Tools are defined as JSON schemas — a list of dicts describing
# each function the LLM can call.
# The LLM reads these descriptions and decides WHEN to call them.
# The description quality directly determines when the LLM calls the tool.

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            # description = what the LLM reads to decide when to use this
            # Be specific — vague descriptions cause wrong tool selection
            "description": "Get the current weather for a specific city. "
                          "Use this when the user asks about weather conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Chennai' or 'San Francisco'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]  # unit is optional
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations. "
                          "Use this for any arithmetic, percentages, or math operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '(15 * 8) + 42'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search an internal knowledge base for information. "
                          "Use this when the user asks about company policies, "
                          "products, or internal documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ============================================================
# SECTION 2: Implementing the actual functions
# ============================================================

# These are the REAL Python functions that run when the LLM calls them.
# The LLM never runs code — it only requests a call.
# Your Python code does the actual execution.

def get_weather(city: str, unit: str = "celsius") -> dict:
    """Simulated weather API call."""
    # In production this would call a real weather API
    weather_data = {
        "Chennai": {"temp": 32, "condition": "humid and partly cloudy"},
        "San Francisco": {"temp": 18, "condition": "foggy"},
        "London": {"temp": 12, "condition": "overcast"},
        "Tokyo": {"temp": 22, "condition": "clear"},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "unknown"})
    temp = data["temp"]
    if unit == "fahrenheit":
        temp = (temp * 9/5) + 32
    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": data["condition"]
    }


def calculate(expression: str) -> dict:
    """Safely evaluate a mathematical expression."""
    try:
        # eval() is dangerous in production — use a proper math parser
        # For learning purposes this is fine
        result = eval(expression, {"__builtins__": {}})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def search_knowledge_base(query: str) -> dict:
    """Simulated knowledge base search."""
    kb = {
        "refund": "Refunds are processed within 5-7 business days.",
        "shipping": "Standard shipping takes 3-5 days. Express is 1-2 days.",
        "pricing": "Our base plan is $29/month. Pro is $99/month.",
        "support": "Support is available 24/7 via email and chat.",
    }
    # Simple keyword matching — in production this would be semantic search
    for keyword, answer in kb.items():
        if keyword.lower() in query.lower():
            return {"query": query, "result": answer, "found": True}
    return {"query": query, "result": "No information found.", "found": False}


# Map function names to actual functions
# This is the dispatch table — how you connect LLM tool calls to real code
AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_knowledge_base": search_knowledge_base,
}


# ============================================================
# SECTION 3: The tool call loop
# ============================================================

# print("\n" + "=" * 50)
# print("SECTION 3: The tool call loop")
# print("=" * 50)

def run_with_tools(user_message: str) -> str:
    """
    Run a conversation turn with tool use.

    The flow:
    1. Send user message + tool definitions to LLM
    2. LLM either responds directly OR requests a tool call
    3. If tool call: execute the function, send result back
    4. LLM uses the result to form its final response

    This loop continues until the LLM gives a direct response.
    """
    messages = [{"role": "user", "content": user_message}]
    print(f"\nUser: {user_message}")

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            # tool_choice="auto" = LLM decides whether to use tools
            # tool_choice="none" = never use tools
            # tool_choice={"type": "function", "function": {"name": "X"}} = force specific tool
            tool_choice="auto",
            max_tokens=500
        )

        message = response.choices[0].message

        # Check if LLM wants to call a tool
        if message.tool_calls:
            print(f"\n[LLM requested {len(message.tool_calls)} tool call(s)]")

            # Add assistant message with tool call request to history
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Execute each requested tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                # arguments come as a JSON string — parse to dict
                arguments = json.loads(tool_call.function.arguments)

                print(f"  Calling: {function_name}({arguments})")

                # Look up and execute the real function
                if function_name in AVAILABLE_FUNCTIONS:
                    result = AVAILABLE_FUNCTIONS[function_name](**arguments)
                else:
                    result = {"error": f"Unknown function: {function_name}"}

                print(f"  Result: {result}")

                # Add tool result to messages
                # role "tool" = this is the output of a tool call
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

            # Loop continues — send results back to LLM for final response

        else:
            # LLM gave a direct response — no more tool calls
            final_response = message.content.strip()
            print(f"\nAssistant: {final_response}")
            return final_response


# ============================================================
# SECTION 4: Testing different tool scenarios
# ============================================================

print("\n" + "=" * 50)
print("SECTION 4: Testing tool scenarios")
print("=" * 50)

# Test 1 — should trigger get_weather
run_with_tools("What's the weather like in SF right now?")

print("\n" + "-" * 40)

# Test 2 — should trigger calculate
run_with_tools("What is 15% of 847, and then add 230 to that?")

print("\n" + "-" * 40)

# Test 3 — should trigger search_knowledge_base
run_with_tools("What's your shipping policy?")

print("\n" + "-" * 40)

# Test 4 — should NOT trigger any tool (direct answer)
run_with_tools("What is the capital of Iran?")

print("\n" + "-" * 40)

# Test 5 — might trigger multiple tools
run_with_tools("What's the weather in Edinburgh and what's 100 divided by 7?")