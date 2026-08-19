"""
NovaTech AI Support Agent - LangGraph Orchestration
====================================================

Day 5 responsibilities:

- Build the LangGraph support workflow
- Provide a public run_agent() API for chat_service
- Preserve Day 1-4 graph behaviour
- Support RAG, tool calling and escalation
- Return a normalized agent result
- Work without requiring OpenAI credits when local mode is enabled
"""

from typing import Any, Dict, List, Optional

from app.agent.nodes import (
    escalate_node,
    rag_node,
    router_node,
    tool_node,
    tool_response_node,
)

from app.agent.state import AgentState


# ============================================================
# ROUTING
# ============================================================

def route_after_router(state: AgentState) -> str:
    """
    Determine the next LangGraph node based on router decision.

    Allowed destinations:

        rag
        tool
        escalate
    """

    action = state.get("action", "escalate")

    if action == "tool":
        return "tool"

    if action == "rag":
        return "rag"

    return "escalate"


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_support_graph():
    """
    Build and compile the NovaTech LangGraph workflow.

    Graph:

        START
          |
        router
          |
      conditional
       /   |   \
     RAG tool escalate
      |    |      |
     END  tool_response
             |
            END
    """

    try:
        from langgraph.graph import END, START, StateGraph

        workflow = StateGraph(AgentState)

        # ----------------------------------------------------
        # Nodes
        # ----------------------------------------------------

        workflow.add_node(
            "router",
            router_node,
        )

        workflow.add_node(
            "rag_node",
            rag_node,
        )

        workflow.add_node(
            "tool_node",
            tool_node,
        )

        workflow.add_node(
            "tool_response_node",
            tool_response_node,
        )

        workflow.add_node(
            "escalate_node",
            escalate_node,
        )

        # ----------------------------------------------------
        # START -> ROUTER
        # ----------------------------------------------------

        workflow.add_edge(
            START,
            "router",
        )

        # ----------------------------------------------------
        # ROUTER -> conditional destination
        # ----------------------------------------------------

        workflow.add_conditional_edges(
            "router",
            route_after_router,
            {
                "rag": "rag_node",
                "tool": "tool_node",
                "escalate": "escalate_node",
            },
        )

        # ----------------------------------------------------
        # Tool pipeline
        # ----------------------------------------------------

        workflow.add_edge(
            "tool_node",
            "tool_response_node",
        )

        workflow.add_edge(
            "tool_response_node",
            END,
        )

        # ----------------------------------------------------
        # Terminal nodes
        # ----------------------------------------------------

        workflow.add_edge(
            "rag_node",
            END,
        )

        workflow.add_edge(
            "escalate_node",
            END,
        )

        return workflow.compile()

    except ImportError:
        """
        Fallback executor.

        This allows the project to continue working even if
        LangGraph is unavailable.
        """

        class CompiledStateGraphFallback:

            def __init__(self):
                self.nodes = {
                    "router": router_node,
                    "rag_node": rag_node,
                    "tool_node": tool_node,
                    "tool_response_node": tool_response_node,
                    "escalate_node": escalate_node,
                }

            def invoke(
                self,
                input_state: AgentState,
            ) -> AgentState:

                state: AgentState = dict(input_state)

                # ------------------------------------------------
                # Router
                # ------------------------------------------------

                router_result = self.nodes["router"](state)

                state.update(router_result)

                # ------------------------------------------------
                # Routing
                # ------------------------------------------------

                destination = route_after_router(state)

                # ------------------------------------------------
                # RAG
                # ------------------------------------------------

                if destination == "rag":

                    rag_result = self.nodes["rag_node"](state)

                    state.update(rag_result)

                # ------------------------------------------------
                # Tools
                # ------------------------------------------------

                elif destination == "tool":

                    tool_result = self.nodes["tool_node"](state)

                    state.update(tool_result)

                    response_result = self.nodes[
                        "tool_response_node"
                    ](state)

                    state.update(response_result)

                # ------------------------------------------------
                # Escalation
                # ------------------------------------------------

                else:

                    escalation_result = self.nodes[
                        "escalate_node"
                    ](state)

                    state.update(escalation_result)

                return state

        return CompiledStateGraphFallback()


# ============================================================
# GLOBAL GRAPH INSTANCE
# ============================================================

support_graph = build_support_graph()


# ============================================================
# PUBLIC AGENT API
# ============================================================

def run_agent(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    conversation_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Public entry point used by chat_service.py.

    Parameters
    ----------
    user_message:
        Current customer message.

    history:
        Previous conversation messages.

    conversation_history:
        Alternative name supported for compatibility.

    conversation_id:
        Optional conversation identifier.

    Returns
    -------
    Dict[str, Any]
        Normalized agent result containing:

        action
        tool_name
        tool_args
        response
        retrieved_documents
        tool_result
        conversation_id
    """

    # --------------------------------------------------------
    # Validate message
    # --------------------------------------------------------

    if user_message is None:
        user_message = ""

    user_message = str(user_message).strip()

    if not user_message:

        return {
            "action": "escalate",
            "tool_name": None,
            "tool_args": {},
            "response": (
                "Please provide a message so I can help you."
            ),
            "retrieved_documents": [],
            "tool_result": None,
            "conversation_id": conversation_id,
        }

    # --------------------------------------------------------
    # Normalize conversation history
    # --------------------------------------------------------

    if conversation_history is not None:

        final_history = conversation_history

    elif history is not None:

        final_history = history

    else:

        final_history = []

    # --------------------------------------------------------
    # Build initial state
    # --------------------------------------------------------

    initial_state: AgentState = {
        "message": user_message,
        "action": None,
        "tool_name": None,
        "tool_args": {},
        "response": None,
        "conversation_history": final_history,
        "retrieved_documents": [],
        "tool_result": None,
    }

    # --------------------------------------------------------
    # Execute LangGraph
    # --------------------------------------------------------

    try:

        final_state = support_graph.invoke(
            initial_state
        )

    except Exception as exc:

        # ----------------------------------------------------
        # Safe failure handling
        # ----------------------------------------------------

        return {
            "action": "escalate",
            "tool_name": None,
            "tool_args": {},
            "response": (
                "I'm unable to process your request right now. "
                "Your request can be handled by human support."
            ),
            "retrieved_documents": [],
            "tool_result": None,
            "conversation_id": conversation_id,
            "error": str(exc),
        }

    # --------------------------------------------------------
    # Normalize returned values
    # --------------------------------------------------------

    action = final_state.get(
        "action",
        "escalate",
    )

    tool_name = final_state.get(
        "tool_name",
    )

    tool_args = final_state.get(
        "tool_args",
        {},
    )

    response = final_state.get(
        "response",
    )

    retrieved_documents = final_state.get(
        "retrieved_documents",
        [],
    )

    tool_result = final_state.get(
        "tool_result",
    )

    # --------------------------------------------------------
    # Ensure response always exists
    # --------------------------------------------------------

    if not response:

        if action == "rag":

            response = (
                "I found relevant information in the "
                "NovaTech knowledge base."
            )

        elif action == "tool":

            response = (
                "I found the requested information."
            )

        else:

            response = (
                "Your request has been flagged for "
                "human support."
            )

    # --------------------------------------------------------
    # Return normalized result
    # --------------------------------------------------------

    return {
        "action": action,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "response": response,
        "retrieved_documents": retrieved_documents,
        "tool_result": tool_result,
        "conversation_id": conversation_id,
    }


# ============================================================
# OPTIONAL SYNCHRONOUS ALIAS
# ============================================================

def invoke_agent(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Compatibility alias.

    Some services/tests may call invoke_agent()
    instead of run_agent().
    """

    return run_agent(
        user_message=user_message,
        history=history,
        **kwargs,
    )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "support_graph",
    "build_support_graph",
    "route_after_router",
    "run_agent",
    "invoke_agent",
]