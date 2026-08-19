import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Offline zero-credit mode
os.environ["EMBEDDING_PROVIDER"] = "local"

from app.agent.graph import build_support_graph
from app.agent.router import RouterDecision
from app.database.db import create_tables, seed_data


def run_day4_manual_demo():
    print("\n========================================================")
    print("      NOVATECH AI SUPPORT AGENT — DAY 4 MANUAL DEMO     ")
    print("========================================================\n")

    create_tables()
    seed_data()
    graph = build_support_graph()

    # DEMO 1: Order Lookup #4521
    print("--------------------------------------------------------")
    print("SCENARIO 1: Live Order Status Lookup")
    print("User Message: \"Where's my order #4521?\"")
    mock_decision_1 = RouterDecision(
        action="tool",
        tool_name="check_order_status",
        args={"order_id": 4521}
    )
    with patch("app.agent.nodes.route_query", return_value=mock_decision_1):
        res1 = graph.invoke({"message": "Where's my order #4521?"})
        print(f"-> Router Action: {res1.get('action')}")
        print(f"-> Tool Selected: {res1.get('tool_name')}")
        print(f"-> Real SQLite Tool Result: {res1.get('tool_result')}")
        print(f"-> Generated Response: {res1.get('response')}\n")

    # DEMO 2: Account Status Lookup #1001
    print("--------------------------------------------------------")
    print("SCENARIO 2: Live Account Status Lookup")
    print("User Message: \"Is account 1001 active?\"")
    mock_decision_2 = RouterDecision(
        action="tool",
        tool_name="check_account_status",
        args={"account_id": 1001}
    )
    with patch("app.agent.nodes.route_query", return_value=mock_decision_2):
        res2 = graph.invoke({"message": "Is account 1001 active?"})
        print(f"-> Router Action: {res2.get('action')}")
        print(f"-> Tool Selected: {res2.get('tool_name')}")
        print(f"-> Real SQLite Tool Result: {res2.get('tool_result')}")
        print(f"-> Generated Response: {res2.get('response')}\n")

    # DEMO 3: Missing Order Lookup #9999
    print("--------------------------------------------------------")
    print("SCENARIO 3: Non-Existent Record Lookup")
    print("User Message: \"Where is order #9999?\"")
    mock_decision_3 = RouterDecision(
        action="tool",
        tool_name="check_order_status",
        args={"order_id": 9999}
    )
    with patch("app.agent.nodes.route_query", return_value=mock_decision_3):
        res3 = graph.invoke({"message": "Where is order #9999?"})
        print(f"-> Router Action: {res3.get('action')}")
        print(f"-> Tool Selected: {res3.get('tool_name')}")
        print(f"-> Real SQLite Tool Result: {res3.get('tool_result')}")
        print(f"-> Generated Response: {res3.get('response')}\n")

    print("========================================================")
    print("         DAY 4 MANUAL DEMO COMPLETED SUCCESSFULLY!      ")
    print("========================================================\n")


if __name__ == "__main__":
    run_day4_manual_demo()
