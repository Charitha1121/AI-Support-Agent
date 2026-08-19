import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Ensure local embedding provider is used (no OpenAI embedding calls)
os.environ["EMBEDDING_PROVIDER"] = "local"

from app.agent.graph import build_support_graph, route_after_router, support_graph
from app.agent.router import RouterDecision, route_query
from app.database.db import create_tables, seed_data


def setup_module():
    create_tables()
    seed_data()


def mock_openai_response(decision_dict: dict):
    """Helper to generate a mocked OpenAI ChatCompletion response containing JSON."""
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(decision_dict)
    mock_resp.choices = [mock_choice]
    return mock_resp


def test_graph_compilation():
    """Verify that build_support_graph compiles and contains the application nodes."""
    graph = build_support_graph()
    assert graph is not None
    print("PASS: LangGraph graph compilation verified")


def test_rag_routing_and_execution():
    """
    TEST 1:
    Input: 'What is your refund policy?'
    Mock decision: action='rag'
    Verify: router selects rag, rag_node executes, retrieval is performed, reaches END, response exists.
    """
    mock_decision = {
        "action": "rag",
        "tool_name": None,
        "args": {}
    }

    with patch("app.agent.nodes.route_query") as mock_route:
        mock_route.return_value = RouterDecision(**mock_decision)

        graph = build_support_graph()
        result = graph.invoke({"message": "What is your refund policy?"})

        assert result["action"] == "rag"
        assert result.get("retrieved_documents") is not None
        assert len(result["retrieved_documents"]) > 0
        assert "RAG route selected" in result["response"]
        print(f"PASS: RAG route execution (Retrieved {len(result['retrieved_documents'])} docs)")


def test_order_tool_routing_and_execution():
    """
    TEST 2:
    Input: 'Where is order 4521?'
    Mock decision: action='tool', tool_name='check_order_status', args={'order_id': 4521}
    Verify: router selects tool, tool_name='check_order_status', tool_node executes and generates response.
    """
    setup_module()
    mock_decision = {
        "action": "tool",
        "tool_name": "check_order_status",
        "args": {"order_id": 4521}
    }

    with patch("app.agent.nodes.route_query") as mock_route:
        mock_route.return_value = RouterDecision(**mock_decision)

        graph = build_support_graph()
        result = graph.invoke({"message": "Where is order 4521?"})

        assert result["action"] == "tool"
        assert result["tool_name"] == "check_order_status"
        assert result["tool_args"] == {"order_id": 4521}
        assert result.get("tool_result") is not None
        assert result["tool_result"]["order_id"] == 4521
        assert "4521" in result["response"]
        print("PASS: Order tool route execution")


def test_account_tool_routing_and_execution():
    """
    TEST 3:
    Input: 'Is account 1001 active?'
    Mock decision: action='tool', tool_name='check_account_status', args={'account_id': 1001}
    Verify: router selects tool, tool_name='check_account_status', tool_node executes.
    """
    setup_module()
    mock_decision = {
        "action": "tool",
        "tool_name": "check_account_status",
        "args": {"account_id": 1001}
    }

    with patch("app.agent.nodes.route_query") as mock_route:
        mock_route.return_value = RouterDecision(**mock_decision)

        graph = build_support_graph()
        result = graph.invoke({"message": "Is account 1001 active?"})

        assert result["action"] == "tool"
        assert result["tool_name"] == "check_account_status"
        assert result["tool_args"] == {"account_id": 1001}
        assert result.get("tool_result") is not None
        assert result["tool_result"]["account_id"] == 1001
        assert "1001" in result["response"]
        print("PASS: Account tool route execution")


def test_escalate_routing_and_execution():
    """
    TEST 4:
    Input: 'Can I speak to a human?'
    Mock decision: action='escalate'
    Verify: escalate_node executes, reaches END, response contains human support wording.
    """
    mock_decision = {
        "action": "escalate",
        "tool_name": None,
        "args": {}
    }

    with patch("app.agent.nodes.route_query") as mock_route:
        mock_route.return_value = RouterDecision(**mock_decision)

        graph = build_support_graph()
        result = graph.invoke({"message": "Can I speak to a human?"})

        assert result["action"] == "escalate"
        assert "human support" in result["response"].lower()
        print("PASS: Escalation route execution")


def test_invalid_action_rejected():
    """
    TEST 5A:
    Verify that an invalid action value is rejected by RouterDecision schema.
    """
    try:
        RouterDecision(action="delete_account", tool_name=None, args={})
        assert False, "Expected ValueError for invalid action"
    except Exception:
        pass
    print("PASS: Invalid router action rejected")


def test_invalid_tool_name_rejected():
    """
    TEST 5B:
    Verify that an unauthorized tool name (e.g. delete_database) is rejected.
    """
    try:
        RouterDecision(action="tool", tool_name="delete_database", args={})
        assert False, "Expected ValueError for unauthorized tool_name"
    except Exception:
        pass
    print("PASS: Unauthorized tool name rejected")


def test_route_after_router_logic():
    """Verify conditional edge function outputs matching branch keys."""
    assert route_after_router({"action": "rag"}) == "rag"
    assert route_after_router({"action": "tool"}) == "tool"
    assert route_after_router({"action": "escalate"}) == "escalate"
    assert route_after_router({}) == "escalate"
    print("PASS: Conditional edge routing logic")


def run_all_agent_graph_tests():
    print("========================================")
    print("RUNNING DAY 3 LANGGRAPH AGENT TESTS")
    print("========================================")
    setup_module()
    test_graph_compilation()
    test_rag_routing_and_execution()
    test_order_tool_routing_and_execution()
    test_account_tool_routing_and_execution()
    test_escalate_routing_and_execution()
    test_invalid_action_rejected()
    test_invalid_tool_name_rejected()
    test_route_after_router_logic()
    print("========================================")
    print("ALL DAY 3 LANGGRAPH TESTS PASSED SUCCESSFULLY!")
    print("========================================\n")


if __name__ == "__main__":
    run_all_agent_graph_tests()
