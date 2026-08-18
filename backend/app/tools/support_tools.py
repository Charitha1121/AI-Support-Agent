from app.database.db import get_connection


def check_order_status(order_id: int) -> dict:
    """
    Look up an order and return its current status.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT order_id, customer_name, status, eta
        FROM orders
        WHERE order_id = ?
    """, (order_id,))

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return {
            "success": False,
            "error": "Order not found"
        }

    return {
        "success": True,
        "order_id": row["order_id"],
        "customer_name": row["customer_name"],
        "status": row["status"],
        "eta": row["eta"]
    }


def check_account_status(account_id: int) -> dict:
    """
    Look up an account and return its current status.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT account_id, plan, renewal_date, status
        FROM accounts
        WHERE account_id = ?
    """, (account_id,))

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return {
            "success": False,
            "error": "Account not found"
        }

    return {
        "success": True,
        "account_id": row["account_id"],
        "plan": row["plan"],
        "renewal_date": row["renewal_date"],
        "status": row["status"]
    }