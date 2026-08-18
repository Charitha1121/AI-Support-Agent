from app.database.db import create_tables, seed_data
from app.tools.support_tools import (
    check_order_status,
    check_account_status,
)


def main():
    # Make sure the database exists and contains data.
    create_tables()
    seed_data()

    print("\n--- ORDER LOOKUP ---")

    order = check_order_status(4521)

    print(order)

    print("\n--- ACCOUNT LOOKUP ---")

    account = check_account_status(1001)

    print(account)

    print("\n--- INVALID ORDER ---")

    invalid_order = check_order_status(9999)

    print(invalid_order)

    print("\n--- INVALID ACCOUNT ---")

    invalid_account = check_account_status(9999)

    print(invalid_account)


if __name__ == "__main__":
    main()