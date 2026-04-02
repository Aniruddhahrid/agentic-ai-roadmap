"""
Day 23: ReAct Loop — Manual Agent from Scratch

ReAct = Reasoning + Acting
The agent loops: think → act → observe → think → act → ...
until it decides to give a final answer.

This is the foundation of all agentic behavior.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment — same pattern from Day 6 onwards
load_dotenv()

# Initialize Ollama client — same OpenAI SDK interface from Day 16 onwards
# We point it at localhost:11434 because Ollama runs locally
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't need a real key, but SDK requires something
)

MODEL = "qwen2.5:7b"

# === TOOL DEFINITIONS (from Day 22) ===

def get_weather(location: str) -> str:
    """
    Fake weather API — in real world this would call OpenWeatherMap or similar.
    Returns a JSON string because that's what LLMs expect from tool results.
    """
    # Simulate API response
    weather_data = {
        "location": location,
        "temperature": "22°C",
        "conditions": "Partly cloudy",
        "humidity": "65%"
    }
    return json.dumps(weather_data)

def calculate(expression: str) -> str:
    """
    Safe calculator using eval() — NEVER use eval() on untrusted input in production.
    This is for learning only. Real agents use sandboxed environments or math parsers.
    """
    try:
        # eval() executes the string as Python code — dangerous but useful for demos
        result = eval(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Tool definitions in OpenAI format — same from Day 22
# This tells the LLM what tools exist and how to call them
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo' or 'London'"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2 + 2' or '15 * 8'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# Dispatch map — same from Day 22
# Maps function names (strings) to actual Python functions
AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate
}

def run_agent(user_query: str, max_iterations: int = 5) -> str:
    """
    The ReAct loop — this is the agent's brain.
    
    Loop structure:
    1. Send messages to LLM
    2. If LLM wants to use a tool → execute it, add result to messages, loop again
    3. If LLM gives text response → that's the final answer, return it
    4. If we hit max_iterations → safety stop (prevents infinite loops)
    
    Why max_iterations? In case the LLM gets confused and keeps calling tools forever.
    Real production agents have cost limits, not just iteration limits.
    """
    # messages list — same conversation history pattern from Day 19
    # We build this up over multiple turns
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Use tools when you need external information. "
                "When you have enough information, respond directly to the user."
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    # The ReAct loop — this is where the magic happens
    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*60}")
        
        # THINK: Ask the LLM what to do next
        # tools=TOOLS tells the LLM what tools it can use
        # The LLM will either call a tool OR give a text response
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS
        )
        
        assistant_message = response.choices[0].message
        
        # Add the LLM's response to our conversation history
        # This is critical — the LLM needs to see its own previous thoughts
        messages.append(assistant_message.model_dump())
        
        # ACT: Check if the LLM wants to use a tool
        if assistant_message.tool_calls:
            # The LLM decided it needs to use a tool
            print(f"\n🤔 REASONING: LLM wants to use tools")
            
            # OBSERVE: Execute each tool call
            # (The LLM can request multiple tools in one turn)
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"\n🔧 ACTION: Calling {function_name} with args: {function_args}")
                
                # Execute the tool — same dispatch pattern from Day 22
                function_to_call = AVAILABLE_FUNCTIONS[function_name]
                function_response = function_to_call(**function_args)
                
                print(f"📊 OBSERVATION: {function_response}")
                
                # Add the tool result to the conversation
                # This is how the LLM "sees" what the tool returned
                # The LLM will use this info in the next iteration
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
            
            # Loop continues — LLM will see the tool results and decide what to do next
            
        else:
            # The LLM gave a text response instead of calling a tool
            # This means it thinks it has enough info to answer
            print(f"\n✅ FINAL ANSWER: LLM responded with text (no more tools needed)")
            print(f"\n{assistant_message.content}")
            return assistant_message.content
    
    # Safety stop — we hit max_iterations
    # This should rarely happen with good prompts
    return f"Agent stopped after {max_iterations} iterations without completing the task."

if __name__ == "__main__":
    # Test Case 1: Single tool use
    print("\n" + "="*80)
    print("TEST 1: Single Tool Use")
    print("="*80)
    result = run_agent("What's the weather in Tokyo?")
    
    print("\n" + "="*80)
    print("TEST 2: Multiple Tool Uses")
    print("="*80)
    # This should call weather, then calculator
    result = run_agent(
        "What's the temperature in Tokyo, and what is that temperature multiplied by 2?"
    )
    
    print("\n" + "="*80)
    print("TEST 3: No Tools Needed")
    print("="*80)
    # This shouldn't use any tools — the LLM knows the answer
    result = run_agent("What is the capital of France?")

    def run_agent_verbose(user_query: str, max_iterations: int = 5) -> str:
        """
        Same as run_agent but prints the full message history at each iteration.
        This shows you EXACTLY what the LLM sees.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to tools."
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
        
        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1} — Messages sent to LLM:")
            print(f"{'='*60}")
            
            # Print what we're sending to the LLM
            # This helps you understand what the LLM "knows" at each step
            for msg in messages:
                role = msg.get("role", "unknown")
                if role == "tool":
                    print(f"  [TOOL RESULT from {msg.get('name')}]: {msg.get('content')[:100]}...")
                elif role == "assistant" and msg.get("tool_calls"):
                    print(f"  [ASSISTANT]: Wants to call {msg['tool_calls'][0]['function']['name']}")
                else:
                    content = msg.get("content", "")
                    print(f"  [{role.upper()}]: {content[:100]}...")
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS
            )
            
            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump())
            
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    function_to_call = AVAILABLE_FUNCTIONS[function_name]
                    function_response = function_to_call(**function_args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_response
                    })
            else:
                print(f"\n✅ FINAL: {assistant_message.content}")
                return assistant_message.content
        
        return f"Stopped after {max_iterations} iterations."

# Test the verbose version
if __name__ == "__main__":
    print("\n" + "="*80)
    print("VERBOSE MODE: See exactly what the LLM sees")
    print("="*80)
    run_agent_verbose("What's the weather (temp) in Tokyo multiplied by 3?")