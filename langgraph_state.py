"""
Day 31: LangGraph State Management

Advanced state patterns:
- State reducers (how to merge state updates)
- Parallel node execution
- State channels
- Checkpointing (save/resume workflows)

This is production-level state management.
"""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from concurrent.futures import ThreadPoolExecutor
import time

# === REDUCER PATTERNS ===

class MultiChannelState(TypedDict):
    """
    State with multiple update strategies.
    
    - messages: ADD to list (accumulate)
    - counter: REPLACE value (overwrite)
    - tags: SET union (combine unique items)
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Accumulate
    counter: int  # Replace
    tags: Annotated[set, operator.or_]  # Union

def demo_reducers():
    """
    Demo: How different reducers work.
    """
    print("\n" + "="*80)
    print("DEMO 1: State Reducers")
    print("="*80)
    
    def node1(state: MultiChannelState) -> MultiChannelState:
        print("\n📝 Node 1 updates:")
        return {
            "messages": [HumanMessage(content="Message from node 1")],
            "counter": 5,
            "tags": {"tag1", "tag2"}
        }
    
    def node2(state: MultiChannelState) -> MultiChannelState:
        print("\n📝 Node 2 updates:")
        return {
            "messages": [AIMessage(content="Message from node 2")],
            "counter": 10,  # This REPLACES the 5 from node1
            "tags": {"tag2", "tag3"}  # Union with node1's tags
        }
    
    workflow = StateGraph(MultiChannelState)
    workflow.add_node("node1", node1)
    workflow.add_node("node2", node2)
    workflow.add_edge("node1", "node2")
    workflow.add_edge("node2", END)
    workflow.set_entry_point("node1")
    
    graph = workflow.compile()
    
    initial_state = {
        "messages": [],
        "counter": 0,
        "tags": set()
    }
    
    final_state = graph.invoke(initial_state)
    
    print("\n" + "="*60)
    print("FINAL STATE:")
    print(f"  Messages: {len(final_state['messages'])} (accumulated from both nodes)")
    print(f"  Counter: {final_state['counter']} (replaced, not added)")
    print(f"  Tags: {final_state['tags']} (union of both sets)")

# === PARALLEL EXECUTION ===

class ParallelState(TypedDict):
    """State for parallel node demo."""
    query: str
    results: Annotated[list, operator.add]

def demo_parallel():
    """
    Demo: Parallel node execution.
    
    Multiple nodes can run simultaneously and all update state.
    This is useful for:
    - Calling multiple APIs at once
    - Running multiple analysis steps
    - Fan-out/fan-in patterns
    """
    print("\n" + "="*80)
    print("DEMO 2: Parallel Node Execution")
    print("="*80)
    
    def source1(state: ParallelState) -> ParallelState:
        """Simulate slow API call."""
        print(f"\n🔍 Source 1 starting...")
        time.sleep(1)  # Simulate API delay
        print(f"✅ Source 1 done")
        return {
            "results": [f"Result from source 1: {state['query']}"]
        }
    
    def source2(state: ParallelState) -> ParallelState:
        """Simulate another slow API call."""
        print(f"\n🔍 Source 2 starting...")
        time.sleep(1)  # Simulate API delay
        print(f"✅ Source 2 done")
        return {
            "results": [f"Result from source 2: {state['query']}"]
        }
    
    def source3(state: ParallelState) -> ParallelState:
        """Third parallel source."""
        print(f"\n🔍 Source 3 starting...")
        time.sleep(1)
        print(f"✅ Source 3 done")
        return {
            "results": [f"Result from source 3: {state['query']}"]
        }
    
    def combine(state: ParallelState) -> ParallelState:
        """Combine all parallel results."""
        print(f"\n🔗 Combining {len(state['results'])} results")
        return {}
    
    workflow = StateGraph(ParallelState)
    
    # Add parallel nodes
    workflow.add_node("source1", source1)
    workflow.add_node("source2", source2)
    workflow.add_node("source3", source3)
    workflow.add_node("combine", combine)
    
    # All sources run in parallel, then combine
    workflow.add_edge("source1", "combine")
    workflow.add_edge("source2", "combine")
    workflow.add_edge("source3", "combine")
    workflow.add_edge("combine", END)
    
    # Set ALL three as entry points (they start simultaneously)
    workflow.set_entry_point("source1")
    workflow.set_entry_point("source2")
    workflow.set_entry_point("source3")
    
    graph = workflow.compile()
    
    start = time.time()
    final_state = graph.invoke({"query": "test query", "results": []})
    elapsed = time.time() - start
    
    print(f"\n⏱️  Total time: {elapsed:.2f}s")
    print(f"💡 Notice: 3 sources with 1s delay each = ~1s total (parallel)")
    print(f"   Sequential would be ~3s")
    print(f"\nFinal results: {final_state['results']}")

# === STATE PERSISTENCE ===

def demo_state_inspection():
    """
    Demo: Inspecting state at each step.
    
    LangGraph tracks state changes — you can see exactly
    what each node modified.
    """
    print("\n" + "="*80)
    print("DEMO 3: State Inspection")
    print("="*80)
    
    class CounterState(TypedDict):
        count: int
        log: Annotated[list, operator.add]
    
    def increment(state: CounterState) -> CounterState:
        new_count = state['count'] + 1
        return {
            "count": new_count,
            "log": [f"Incremented to {new_count}"]
        }
    
    def double(state: CounterState) -> CounterState:
        new_count = state['count'] * 2
        return {
            "count": new_count,
            "log": [f"Doubled to {new_count}"]
        }
    
    workflow = StateGraph(CounterState)
    workflow.add_node("increment", increment)
    workflow.add_node("double", double)
    workflow.add_node("increment2", increment)
    
    workflow.add_edge("increment", "double")
    workflow.add_edge("double", "increment2")
    workflow.add_edge("increment2", END)
    workflow.set_entry_point("increment")
    
    graph = workflow.compile()
    
    initial_state = {"count": 0, "log": []}
    
    # stream() gives you state after each node
    print("\n📊 State evolution:")
    for i, state in enumerate(graph.stream(initial_state)):
        print(f"\nStep {i + 1}:")
        print(f"  Count: {list(state.values())[0]['count']}")
        print(f"  Log: {list(state.values())[0]['log']}")

# === MAIN ===

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DAY 31: LangGraph State Management")
    print("="*80)
    
    demo_reducers()
    demo_parallel()
    demo_state_inspection()
    
    print("\n" + "="*80)
    print("KEY CONCEPTS")
    print("="*80)
    print("""
1. Reducers control how state updates merge:
   - operator.add = accumulate (lists)
   - Default = replace (simple values)
   - operator.or_ = union (sets)

2. Parallel execution:
   - Multiple entry points = nodes run simultaneously
   - All converge at next shared node
   - Faster than sequential for I/O-bound tasks

3. State inspection:
   - stream() gives state after each node
   - Useful for debugging complex workflows

Week 5 complete! Next week (Week 6): MCP + FastAPI
Eagle building starts tomorrow (30 min/day + weekends)
    """)