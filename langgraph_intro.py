"""
Day 29: LangGraph Introduction

LangGraph = graph-based agent workflows
- Nodes = steps in your workflow (LLM call, tool use, logic)
- Edges = transitions between steps
- State = shared data structure passed through the graph
- Conditional edges = "if X then go to Y, else go to Z"

Why LangGraph vs manual loops (Week 4)?
- Cleaner code for complex workflows
- Built-in state management
- Easy to visualize and debug
- Handles cycles, conditionals, parallel execution
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import operator

# === BASIC CONCEPT: STATE ===

class AgentState(TypedDict):
    """
    State = the data structure passed between nodes.
    
    Every node receives the current state and returns an updated state.
    Think of it like Redux/Zustand in frontend — centralized state management.
    
    Annotated[list, operator.add] means:
    - When multiple nodes update 'messages', ADD to the list instead of replacing
    - This is how LangGraph handles accumulation (like conversation history)
    """
    messages: Annotated[list, operator.add]
    iteration_count: int
    user_query: str

# === NODES: THE WORK HAPPENS HERE ===

def input_node(state: AgentState) -> AgentState:
    """
    Node 1: Handle user input.
    
    Nodes are just Python functions that:
    1. Take state as input
    2. Do some work
    3. Return updated state
    
    This node initializes the conversation with the user's query.
    """
    print(f"\n📥 INPUT NODE")
    print(f"   User query: {state['user_query']}")
    
    # Add user message to the conversation
    # Note: We return a dict with ONLY the fields we want to update
    # LangGraph merges this with existing state
    return {
        "messages": [HumanMessage(content=state['user_query'])],
        "iteration_count": 0
    }

def thinking_node(state: AgentState) -> AgentState:
    """
    Node 2: Simulate LLM thinking.
    
    In real workflow, this would call an LLM.
    For demo, we just add an AI message.
    """
    print(f"\n🤔 THINKING NODE (Iteration {state['iteration_count'] + 1})")
    
    # Simulate LLM response
    response = AIMessage(content=f"I'm thinking about: {state['user_query']}")
    
    return {
        "messages": [response],
        "iteration_count": state['iteration_count'] + 1
    }

def action_node(state: AgentState) -> AgentState:
    """
    Node 3: Take action based on thinking.
    
    In real workflow, this might call tools.
    """
    print(f"\n⚡ ACTION NODE")
    print(f"   Taking action based on query...")
    
    action_msg = AIMessage(content=f"Action completed for: {state['user_query']}")
    
    return {
        "messages": [action_msg]
    }

def output_node(state: AgentState) -> AgentState:
    """
    Node 4: Format final output.
    """
    print(f"\n📤 OUTPUT NODE")
    print(f"   Total messages: {len(state['messages'])}")
    print(f"   Total iterations: {state['iteration_count']}")
    
    final_msg = AIMessage(content="Workflow complete!")
    
    return {
        "messages": [final_msg]
    }

# === CONDITIONAL ROUTING ===

def should_continue(state: AgentState) -> str:
    """
    Conditional edge: Decide where to go next.
    
    This is the key difference from Week 4's linear loops.
    Based on state, we can route to different nodes.
    
    Returns:
    - "continue": Go to thinking_node again
    - "end": Go to output_node and finish
    """
    # If we've iterated less than 3 times, keep thinking
    if state['iteration_count'] < 3:
        print(f"\n🔄 ROUTER: Continue iterating ({state['iteration_count']}/3)")
        return "continue"
    else:
        print(f"\n✅ ROUTER: Done iterating, move to output")
        return "end"

# === BUILD THE GRAPH ===

def create_simple_graph():
    """
    Create a LangGraph workflow.
    
    Steps:
    1. Create StateGraph with our state schema
    2. Add nodes (the work functions)
    3. Add edges (the transitions)
    4. Set entry point (where to start)
    5. Compile the graph
    """
    # Initialize graph with our state type
    workflow = StateGraph(AgentState)
    
    # Add nodes
    # Syntax: add_node(name, function)
    workflow.add_node("input", input_node)
    workflow.add_node("thinking", thinking_node)
    workflow.add_node("action", action_node)
    workflow.add_node("output", output_node)
    
    # Add edges (transitions)
    # Simple edge: always go from A to B
    workflow.add_edge("input", "thinking")
    workflow.add_edge("thinking", "action")
    
    # Conditional edge: route based on function return value
    # Syntax: add_conditional_edges(
    #   from_node,
    #   condition_function,
    #   {return_value: destination_node}
    # )
    workflow.add_conditional_edges(
        "action",
        should_continue,
        {
            "continue": "thinking",  # Loop back to thinking
            "end": "output"  # Move to output
        }
    )
    
    # Output node always ends the workflow
    workflow.add_edge("output", END)
    
    # Set where the graph starts
    workflow.set_entry_point("input")
    
    # Compile into executable graph
    graph = workflow.compile()
    
    return graph

# === RUN THE GRAPH ===

def run_simple_workflow():
    """
    Execute the graph with initial state.
    """
    print("\n" + "="*80)
    print("DEMO 1: Simple LangGraph Workflow")
    print("="*80)
    
    graph = create_simple_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "iteration_count": 0,
        "user_query": "What's the weather in Tokyo?"
    }
    
    # Run the graph
    # invoke() executes the entire workflow
    final_state = graph.invoke(initial_state)
    
    print("\n" + "="*80)
    print("FINAL STATE")
    print("="*80)
    print(f"Total messages: {len(final_state['messages'])}")
    print(f"Iterations: {final_state['iteration_count']}")
    
    print("\nMessage history:")
    for i, msg in enumerate(final_state['messages'], 1):
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"{i}. [{role}] {msg.content}")

# === COMPARISON: WEEK 4 vs LANGGRAPH ===

def week4_style_loop():
    """
    For comparison: How we'd do this in Week 4 style.
    
    Notice how much manual state management we need.
    """
    print("\n" + "="*80)
    print("COMPARISON: Week 4 Manual Loop Style")
    print("="*80)
    
    # Manual state tracking
    messages = []
    iteration_count = 0
    user_query = "What's the weather in Tokyo?"
    
    # Manual step execution
    print(f"\n📥 Manual: User input")
    messages.append({"role": "user", "content": user_query})
    
    # Manual loop
    while iteration_count < 3:
        print(f"\n🤔 Manual: Thinking (iteration {iteration_count + 1})")
        messages.append({"role": "ai", "content": f"Thinking about: {user_query}"})
        
        print(f"\n⚡ Manual: Action")
        messages.append({"role": "ai", "content": "Action completed"})
        
        iteration_count += 1
    
    print(f"\n📤 Manual: Output")
    messages.append({"role": "ai", "content": "Done"})
    
    print(f"\n   Total messages: {len(messages)}")
    print(f"   Iterations: {iteration_count}")
    
    print("\n💡 NOTICE:")
    print("   Week 4 style: Manual tracking, hardcoded flow, messy for complex logic")
    print("   LangGraph: Declarative nodes, clean state, easy conditional routing")

# === MAIN ===

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DAY 29: LangGraph Introduction")
    print("="*80)
    
    # Run LangGraph version
    run_simple_workflow()
    
    # Show Week 4 comparison
    week4_style_loop()
    
    print("\n" + "="*80)
    print("KEY CONCEPTS")
    print("="*80)
    print("""
1. State = shared data structure passed through the graph
2. Nodes = functions that take state, do work, return updated state
3. Edges = transitions between nodes (simple or conditional)
4. Conditional routing = different paths based on state
5. Much cleaner than manual loops for complex workflows

Tomorrow (Day 30): Build a real agent with LangGraph (tools, LLM, conditional logic)
    """)