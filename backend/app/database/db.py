import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parent / "support.db"


def get_connection():
    """Create and return a connection to the SQLite database."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_tables():
    """Create the orders and accounts tables."""
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            status TEXT NOT NULL,
            eta TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            account_id INTEGER PRIMARY KEY,
            plan TEXT NOT NULL,
            renewal_date TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def seed_data():
    """Insert fake support data into the database."""
    connection = get_connection()
    cursor = connection.cursor()

    orders = [
        (4521, "Rahul", "Shipped", "2026-08-20"),
        (4522, "Priya", "Processing", "2026-08-22"),
        (4523, "Arjun", "Delivered", "2026-08-17"),
        (4524, "Sneha", "Out for delivery", "2026-08-18"),
        (4525, "Kiran", "Cancelled", "N/A"),
        (4526, "Ananya", "Processing", "2026-08-23"),
        (4527, "Vikram", "Shipped", "2026-08-21"),
        (4528, "Meena", "Delivered", "2026-08-16"),
        (4529, "Ravi", "Shipped", "2026-08-24"),
        (4530, "Divya", "Processing", "2026-08-25"),
    ]

    accounts = [
        (1001, "Pro", "2026-09-12", "Active"),
        (1002, "Basic", "2026-08-25", "Active"),
        (1003, "Premium", "2026-10-04", "Suspended"),
        (1004, "Pro", "2026-09-30", "Active"),
        (1005, "Basic", "2026-08-20", "Cancelled"),
        (1006, "Premium", "2026-11-15", "Active"),
        (1007, "Pro", "2026-09-05", "Active"),
        (1008, "Basic", "2026-12-01", "Active"),
        (1009, "Premium", "2026-10-22", "Active"),
        (1010, "Pro", "2026-08-30", "Suspended"),
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO orders
        (order_id, customer_name, status, eta)
        VALUES (?, ?, ?, ?)
    """, orders)

    cursor.executemany("""
        INSERT OR IGNORE INTO accounts
        (account_id, plan, renewal_date, status)
        VALUES (?, ?, ?, ?)
    """, accounts)

    connection.commit()
    connection.close()


if __name__ == "__main__":
    create_tables()
    seed_data()
    print("Database created and seeded successfully.")