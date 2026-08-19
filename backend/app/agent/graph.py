from typing import Any, Dict, Optional

from app.agent.nodes import escalate_node, rag_node, router_node, tool_node, tool_response_node
from app.agent.state import AgentState


def route_after_router(state: AgentState) -> str:
    """
    Inspect the action classified by the router node and determine the next destination.
    """
    action = state.get("action", "escalate")
    if action == "tool":
        return "tool"
    elif action == "rag":
        return "rag"
    return "escalate"


def build_support_graph():
    """
    Construct, wire, and compile the LangGraph agent graph for NovaTech customer support (Day 4).
    
    Graph Topology:
    START -> router -> conditional_edges:
                         ├── rag -> rag_node -> END
                         ├── tool -> tool_node -> tool_response_node -> END
                         └── escalate -> escalate_node -> END
    """
    try:
        from langgraph.graph import END, START, StateGraph

        workflow = StateGraph(AgentState)

        # 1. Add application nodes
        workflow.add_node("router", router_node)
        workflow.add_node("rag_node", rag_node)
        workflow.add_node("tool_node", tool_node)
        workflow.add_node("tool_response_node", tool_response_node)
        workflow.add_node("escalate_node", escalate_node)

        # 2. Add entry point and conditional routing
        workflow.add_edge(START, "router")
        workflow.add_conditional_edges(
            "router",
            route_after_router,
            {
                "rag": "rag_node",
                "tool": "tool_node",
                "escalate": "escalate_node",
            }
        )

        # 3. Add tool execution pipeline and terminal edges to END
        workflow.add_edge("tool_node", "tool_response_node")
        workflow.add_edge("tool_response_node", END)
        workflow.add_edge("rag_node", END)
        workflow.add_edge("escalate_node", END)

        return workflow.compile()

    except ImportError:
        # Fallback executor adhering to the exact StateGraph contract
        class CompiledStateGraphFallback:
            def __init__(self):
                self.nodes = {
                    "router": router_node,
                    "rag_node": rag_node,
                    "tool_node": tool_node,
                    "tool_response_node": tool_response_node,
                    "escalate_node": escalate_node,
                }

            def invoke(self, input_state: AgentState) -> AgentState:
                state: AgentState = dict(input_state)
                # 1. Router node execution
                router_res = self.nodes["router"](state)
                state.update(router_res)

                # 2. Conditional edge routing
                destination = route_after_router(state)
                if destination == "tool":
                    # Tool execution followed by response generation
                    tool_res = self.nodes["tool_node"](state)
                    state.update(tool_res)
                    resp_res = self.nodes["tool_response_node"](state)
                    state.update(resp_res)
                elif destination == "rag":
                    rag_res = self.nodes["rag_node"](state)
                    state.update(rag_res)
                else:
                    esc_res = self.nodes["escalate_node"](state)
                    state.update(esc_res)

                return state

        return CompiledStateGraphFallback()


# Pre-built agent graph instance
support_graph = build_support_graph()
