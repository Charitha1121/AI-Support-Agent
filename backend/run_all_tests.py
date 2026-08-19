import os
import sys
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Default EMBEDDING_PROVIDER to 'local' for zero-credit testing
os.environ.setdefault("EMBEDDING_PROVIDER", "local")

from tests.test_tools import run_all_tool_tests
from tests.test_rag import run_all_rag_tests
from tests.test_router import run_all_router_tests
from tests.test_agent_graph import run_all_agent_graph_tests
from tests.test_day4_tools import run_all_day4_tests


def main():
    print("\n==========================================================")
    print("  NOVATECH AI SUPPORT AGENT — TEST SUITE (DAYS 1, 2, 3, 4)")
    print("==========================================================\n")

    # 1. Day 1: Tools
    run_all_tool_tests()

    # 2. Day 2: Knowledge Base & RAG
    run_all_rag_tests()

    # 3. Day 3: Router & LangGraph Agent Skeleton
    run_all_router_tests()
    run_all_agent_graph_tests()

    # 4. Day 4: Real Database Tool-Calling Integration
    run_all_day4_tests()

    print("==========================================================")
    print(" ALL DAY 1, DAY 2, DAY 3 & DAY 4 TESTS PASSED (100% PASS) ")
    print("==========================================================\n")


if __name__ == "__main__":
    main()
