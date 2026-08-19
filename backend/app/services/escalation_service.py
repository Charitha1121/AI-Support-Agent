from typing import Any, Dict, List, Optional
from app.database.db import get_connection


def list_escalations(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent escalation tickets logged in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, conversation_id, user_message, reason, timestamp, status
        FROM escalations
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "user_message": row["user_message"],
            "reason": row["reason"],
            "timestamp": row["timestamp"],
            "status": row["status"],
        }
        for row in rows
    ]


def get_escalation(escalation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single escalation by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, conversation_id, user_message, reason, timestamp, status
        FROM escalations
        WHERE id = ?
    """, (escalation_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "user_message": row["user_message"],
        "reason": row["reason"],
        "timestamp": row["timestamp"],
        "status": row["status"],
    }
