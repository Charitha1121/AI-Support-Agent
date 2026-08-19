import os
import sys

from pathlib import Path


backend_dir = Path(__file__).resolve().parent

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


# ZERO-CREDIT MODE
os.environ.setdefault(
    "EMBEDDING_PROVIDER",
    "local",
)

os.environ.setdefault(
    "ENABLE_OPENAI_ROUTER",
    "false",
)


from tests.test_tools import run_all_tool_tests
from tests.test_rag import run_all_rag_tests
from tests.test_router import run_all_router_tests
from tests.test_agent_graph import run_all_agent_graph_tests
from tests.test_day4_tools import run_all_day4_tests
from tests.test_day5_api import run_all_day5_tests


def main():

    print("\n==========================================================")
    print(" NOVATECH AI SUPPORT AGENT — TEST SUITE (DAYS 1–5)")
    print("==========================================================\n")


    print("========================================")
    print("RUNNING DAY 1 TOOL TESTS")
    print("========================================")

    run_all_tool_tests()


    print("\n========================================")
    print("RUNNING DAY 2 KNOWLEDGE BASE & RAG TESTS")
    print("========================================")

    run_all_rag_tests()


    print("\n========================================")
    print("RUNNING DAY 3 ROUTER TESTS")
    print("========================================")

    run_all_router_tests()

    run_all_agent_graph_tests()


    print("\n========================================")
    print("RUNNING DAY 4 REAL TOOL-CALLING TESTS")
    print("========================================")

    run_all_day4_tests()


    print("\n========================================")
    print("RUNNING DAY 5 API & MEMORY TESTS")
    print("========================================")

    run_all_day5_tests()


    print("\n==========================================================")
    print("       ALL DAY 1–5 TESTS PASSED SUCCESSFULLY!")
    print("==========================================================\n")


if __name__ == "__main__":
    main()