import sys
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database.db import create_tables, seed_data
from app.tools.support_tools import check_account_status, check_order_status


def test_order_lookup_valid():
    create_tables()
    seed_data()
    order = check_order_status(4521)
    assert order["success"] is True
    assert order["order_id"] == 4521
    assert order["customer_name"] == "Rahul"
    assert order["status"] == "Shipped"
    assert "eta" in order
    print("PASS: Valid order lookup (4521)")


def test_order_lookup_invalid():
    create_tables()
    seed_data()
    order = check_order_status(9999)
    assert order["success"] is False
    assert "error" in order
    print("PASS: Invalid order lookup (9999)")


def test_account_lookup_valid():
    create_tables()
    seed_data()
    account = check_account_status(1001)
    assert account["success"] is True
    assert account["account_id"] == 1001
    assert account["plan"] == "Pro"
    assert account["status"] == "Active"
    assert "renewal_date" in account
    print("PASS: Valid account lookup (1001)")


def test_account_lookup_invalid():
    create_tables()
    seed_data()
    account = check_account_status(9999)
    assert account["success"] is False
    assert "error" in account
    print("PASS: Invalid account lookup (9999)")


def run_all_tool_tests():
    print("========================================")
    print("RUNNING DAY 1 TOOL TESTS")
    print("========================================")
    test_order_lookup_valid()
    test_order_lookup_invalid()
    test_account_lookup_valid()
    test_account_lookup_invalid()
    print("========================================")
    print("ALL DAY 1 TOOL TESTS PASSED!")
    print("========================================\n")


if __name__ == "__main__":
    run_all_tool_tests()
