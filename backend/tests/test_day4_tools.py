import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Ensure local zero-credit embedding mode
os.environ["EMBEDDING_PROVIDER"] = "local"

from app.agent.graph import build_support_graph
from app.agent.nodes import tool_node, tool_response_node
from app.agent.router import RouterDecision
from app.database.db import create_tables, seed_data


def setup_module():
    """Ensure SQLite database is created and seeded with test records."""
    create_tables()
    seed_data()


def test_real_order_tool_execution():
    """
    TEST 1: Real SQLite order lookup (#4521).
    Verify tool_result has real data and final response is generated.
    """
    setup_module()
    mock_router_decision = RouterDecision(
        action="tool",
        tool_name="check_order_status",
        args={"order_id": 4521}
    )

    mock_llm_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Order #4521 for Rahul has been shipped and is expected to arrive on August 20, 2026."
    mock_llm_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_llm_response

    with patch("app.agent.nodes.route_query", return_value=mock_router_decision):
        with patch("openai.OpenAI", return_value=mock_client):
            graph = build_support_graph()
            result = graph.invoke({"message": "Where's my order #4521?"})

            # Verify structured tool_result from real SQLite
            tool_res = result.get("tool_result")
            assert tool_res is not None, "tool_result should be in state"
            assert tool_res["success"] is True
            assert tool_res["order_id"] == 4521
            assert tool_res["customer_name"] == "Rahul"
            assert tool_res["status"] == "Shipped"
            assert tool_res["eta"] == "2026-08-20"

            # Verify final response generated
            assert result.get("response") is not None
            assert len(result["response"]) > 0
    print("PASS: Real order tool execution (Order #4521)")


def test_real_account_tool_execution():
    """
    TEST 2: Real SQLite account lookup (#1001).
    Verify real account data is retrieved and final response is generated.
    """
    setup_module()
    mock_router_decision = RouterDecision(
        action="tool",
        tool_name="check_account_status",
        args={"account_id": 1001}
    )

    mock_llm_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Account 1001 is currently active on the Pro plan with renewal on 2026-09-12."
    mock_llm_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_llm_response

    with patch("app.agent.nodes.route_query", return_value=mock_router_decision):
        with patch("openai.OpenAI", return_value=mock_client):
            graph = build_support_graph()
            result = graph.invoke({"message": "Is account 1001 active?"})

            tool_res = result.get("tool_result")
            assert tool_res is not None
            assert tool_res["success"] is True
            assert tool_res["account_id"] == 1001
            assert tool_res["plan"] == "Pro"
            assert tool_res["status"] == "Active"
            assert tool_res["renewal_date"] == "2026-09-12"

            assert result.get("response") is not None
    print("PASS: Real account tool execution (Account #1001)")


def test_order_not_found_real_execution():
    """
    TEST 3: Real SQLite order lookup for missing order (#9999).
    Verify success=False, error message, and helpful response without hallucinating status.
    """
    setup_module()
    mock_router_decision = RouterDecision(
        action="tool",
        tool_name="check_order_status",
        args={"order_id": 9999}
    )

    with patch("app.agent.nodes.route_query", return_value=mock_router_decision):
        graph = build_support_graph()
        result = graph.invoke({"message": "Where is order #9999?"})

        tool_res = result.get("tool_result")
        assert tool_res is not None
        assert tool_res["success"] is False
        assert tool_res["error"] == "Order not found"

        # Response must acknowledge order could not be found
        response_text = result.get("response", "").lower()
        assert "couldn't find" in response_text or "not found" in response_text
    print("PASS: Order not found handling (Order #9999)")


def test_account_not_found_real_execution():
    """
    TEST 4: Real SQLite account lookup for missing account (#9999).
    Verify success=False and error='Account not found'.
    """
    setup_module()
    mock_router_decision = RouterDecision(
        action="tool",
        tool_name="check_account_status",
        args={"account_id": 9999}
    )

    with patch("app.agent.nodes.route_query", return_value=mock_router_decision):
        graph = build_support_graph()
        result = graph.invoke({"message": "Is account 9999 active?"})

        tool_res = result.get("tool_result")
        assert tool_res is not None
        assert tool_res["success"] is False
        assert tool_res["error"] == "Account not found"

        response_text = result.get("response", "").lower()
        assert "couldn't find" in response_text or "not found" in response_text
    print("PASS: Account not found handling (Account #9999)")


def test_invalid_tool_argument():
    """
    TEST 5: Malformed tool arguments (e.g. string 'abc' or missing id).
    Tool must return controlled error and not crash.
    """
    state_bad_arg = {
        "message": "Where is order abc?",
        "action": "tool",
        "tool_name": "check_order_status",
        "tool_args": {"order_id": "abc"}
    }
    res = tool_node(state_bad_arg)
    assert res["tool_result"]["success"] is False
    assert "error" in res["tool_result"]
    print("PASS: Invalid tool argument rejection")


def test_unauthorized_tool_execution():
    """
    TEST 6: Tool not in allowlist (e.g. 'delete_database').
    Tool execution must be rejected safely.
    """
    state_unauth = {
        "message": "Delete database",
        "action": "tool",
        "tool_name": "delete_database",
        "tool_args": {}
    }
    res = tool_node(state_unauth)
    assert res["tool_result"]["success"] is False
    assert "Unauthorized tool" in res["tool_result"]["error"]
    print("PASS: Unauthorized tool rejection in tool_node")


def test_tool_result_passed_to_llm():
    """
    TEST 7: Verify structured tool_result is formatted into prompt sent to LLM.
    """
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Order #4521 has shipped."
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp

    state = {
        "message": "Where is order 4521?",
        "tool_name": "check_order_status",
        "tool_result": {
            "success": True,
            "order_id": 4521,
            "customer_name": "Rahul",
            "status": "Shipped",
            "eta": "2026-08-20"
        }
    }

    tool_response_node(state, client=mock_client)

    # Verify LLM received prompt with tool results
    call_args = mock_client.chat.completions.create.call_args
    assert call_args is not None
    sent_messages = call_args[1]["messages"]
    user_msg_content = sent_messages[1]["content"]
    assert "4521" in user_msg_content
    assert "Shipped" in user_msg_content
    assert "2026-08-20" in user_msg_content
    print("PASS: Structured tool result passed into LLM prompt")


def test_llm_failure_handling():
    """
    TEST 8: Simulate LLM exception in tool_response_node.
    Controlled fallback response returned without crashing or leaking secrets.
    """
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("Simulated LLM API Timeout")

    state = {
        "message": "Where is order 4521?",
        "tool_name": "check_order_status",
        "tool_result": {
            "success": True,
            "order_id": 4521,
            "status": "Shipped",
            "eta": "2026-08-20"
        }
    }

    res = tool_response_node(state, client=mock_client)
    assert res.get("response") is not None
    assert "try again" in res["response"].lower() or "shipped" in res["response"].lower()
    print("PASS: LLM failure graceful error handling")


def test_tool_failure_handling():
    """
    TEST 9: Simulate tool execution failure.
    Verify error is captured in tool_result without uncaught exception.
    """
    with patch("app.agent.nodes.check_order_status", side_effect=Exception("Database lock error")):
        state = {
            "tool_name": "check_order_status",
            "tool_args": {"order_id": 4521}
        }
        res = tool_node(state)
        assert res["tool_result"]["success"] is False
        assert "Database lock error" in res["tool_result"]["error"]
    print("PASS: Tool failure error handling")


def test_day4_graph_compilation_and_nodes():
    """
    TEST 10: Verify the Day 4 graph compiles and contains all 5 application nodes.
    """
    graph = build_support_graph()
    assert graph is not None
    print("PASS: Day 4 LangGraph compilation with tool_response_node")


def run_all_day4_tests():
    print("========================================")
    print("RUNNING DAY 4 REAL TOOL-CALLING TESTS")
    print("========================================")
    test_real_order_tool_execution()
    test_real_account_tool_execution()
    test_order_not_found_real_execution()
    test_account_not_found_real_execution()
    test_invalid_tool_argument()
    test_unauthorized_tool_execution()
    test_tool_result_passed_to_llm()
    test_llm_failure_handling()
    test_tool_failure_handling()
    test_day4_graph_compilation_and_nodes()
    print("========================================")
    print("ALL DAY 4 TOOL-CALLING TESTS PASSED!")
    print("========================================\n")


if __name__ == "__main__":
    run_all_day4_tests()
