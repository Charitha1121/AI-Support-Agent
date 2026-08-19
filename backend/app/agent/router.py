"""
NovaTech AI Support Agent
Local / OpenAI Query Router
"""

import json
import os
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================
# ROUTER DECISION
# ============================================================

class RouterDecision(BaseModel):
    """
    Strict routing decision used by the LangGraph agent.
    """

    action: Literal[
        "rag",
        "tool",
        "escalate",
    ]

    tool_name: Optional[
        Literal[
            "check_order_status",
            "check_account_status",
        ]
    ] = None

    args: Dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:

        if value not in {
            "rag",
            "tool",
            "escalate",
        }:
            raise ValueError(
                f"Invalid action: {value}"
            )

        return value

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(
        cls,
        value: Optional[str],
    ) -> Optional[str]:

        allowed = {
            "check_order_status",
            "check_account_status",
        }

        if value is not None and value not in allowed:
            raise ValueError(
                f"Unauthorized tool: {value}"
            )

        return value


# ============================================================
# OPENAI ROUTER PROMPT
# ============================================================

ROUTER_SYSTEM_PROMPT = """
You are the NovaTech Customer Support routing agent.

Classify the user request into exactly one category:

1. rag
   General knowledge or company policy questions.

2. tool
   Questions requiring order/account information.

3. escalate
   Human support, complaints, unsupported requests.

Allowed tools:

check_order_status
check_account_status

Return valid JSON only.
"""


# ============================================================
# LOCAL ROUTER
# ============================================================

def local_route_query(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> RouterDecision:
    """
    Zero-cost deterministic router.

    This does NOT use OpenAI.

    It is intentionally simple and predictable so that:

    - Day 3 tests pass
    - Day 5 API tests pass
    - Development works without API credits
    - Tool access remains allowlisted
    """

    message = (
        user_message or ""
    ).strip().lower()

    # --------------------------------------------------------
    # Human escalation
    # --------------------------------------------------------

    escalation_patterns = [
        "speak with a human",
        "talk to a human",
        "human agent",
        "human support",
        "customer representative",
        "customer rep",
        "speak to an agent",
        "talk to an agent",
        "supervisor",
        "manager",
        "real person",
        "complaint",
    ]

    if any(
        pattern in message
        for pattern in escalation_patterns
    ):
        return RouterDecision(
            action="escalate",
            tool_name=None,
            args={},
        )

    # --------------------------------------------------------
    # Order tool
    # --------------------------------------------------------

    order_keywords = [
        "order",
        "shipment",
        "tracking",
        "track my package",
        "where is my package",
        "delivery status",
    ]

    if any(
        keyword in message
        for keyword in order_keywords
    ):

        # Find integer IDs such as 4521
        numbers = re.findall(
            r"\b\d{3,10}\b",
            message,
        )

        if numbers:

            order_id = int(numbers[0])

            return RouterDecision(
                action="tool",
                tool_name="check_order_status",
                args={
                    "order_id": order_id
                },
            )

        # Order-specific request without ID
        return RouterDecision(
            action="escalate",
            tool_name=None,
            args={},
        )

    # --------------------------------------------------------
    # Account tool
    # --------------------------------------------------------

    account_keywords = [
        "account",
        "my account",
        "account status",
        "account plan",
        "renewal",
        "subscription status",
    ]

    if any(
        keyword in message
        for keyword in account_keywords
    ):

        numbers = re.findall(
            r"\b\d{3,10}\b",
            message,
        )

        if numbers:

            account_id = int(numbers[0])

            return RouterDecision(
                action="tool",
                tool_name="check_account_status",
                args={
                    "account_id": account_id
                },
            )

        return RouterDecision(
            action="escalate",
            tool_name=None,
            args={},
        )

    # --------------------------------------------------------
    # RAG / knowledge base
    # --------------------------------------------------------

    rag_keywords = [
        "refund",
        "refund policy",
        "return",
        "return policy",
        "shipping",
        "delivery",
        "subscription",
        "subscription plans",
        "pricing",
        "price",
        "cancel",
        "cancellation",
        "password",
        "security",
        "payment",
        "payment methods",
        "support",
        "contact",
        "support hours",
        "business hours",
        "policy",
        "how long",
        "how much",
        "what do you offer",
    ]

    if any(
        keyword in message
        for keyword in rag_keywords
    ):
        return RouterDecision(
            action="rag",
            tool_name=None,
            args={},
        )

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return RouterDecision(
        action="escalate",
        tool_name=None,
        args={},
    )


# ============================================================
# MAIN ROUTER
# ============================================================
def route_query(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    client: Optional[Any] = None,
) -> RouterDecision:

    # =========================================================
    # ZERO-CREDIT / LOCAL ROUTER
    # =========================================================

    enable_openai = (
        os.getenv(
            "ENABLE_OPENAI_ROUTER",
            "false",
        )
        .strip()
        .lower()
        == "true"
    )

    if not enable_openai:

        text = user_message.lower()

        # -----------------------------------------------------
        # Order lookup
        # -----------------------------------------------------

        import re

        order_match = re.search(
            r"\border\s*#?\s*(\d+)\b",
            text,
        )

        if order_match:

            order_id = int(
                order_match.group(1)
            )

            return RouterDecision(
                action="tool",
                tool_name="check_order_status",
                args={
                    "order_id": order_id
                },
            )

        # -----------------------------------------------------
        # Account lookup
        # -----------------------------------------------------

        account_match = re.search(
            r"\baccount\s*#?\s*(\d+)\b",
            text,
        )

        if account_match:

            account_id = int(
                account_match.group(1)
            )

            return RouterDecision(
                action="tool",
                tool_name="check_account_status",
                args={
                    "account_id": account_id
                },
            )

        # -----------------------------------------------------
        # Human escalation
        # -----------------------------------------------------

        escalation_keywords = [
            "speak with a human",
            "talk to a human",
            "human agent",
            "customer representative",
            "representative",
            "supervisor",
            "human support",
            "real person",
        ]

        if any(
            keyword in text
            for keyword in escalation_keywords
        ):

            return RouterDecision(
                action="escalate",
                tool_name=None,
                args={},
            )

        # -----------------------------------------------------
        # RAG / knowledge-base questions
        # -----------------------------------------------------

        rag_keywords = [
            "refund",
            "return",
            "shipping",
            "delivery",
            "subscription",
            "plan",
            "pricing",
            "cancel",
            "cancellation",
            "password",
            "security",
            "payment",
            "support",
            "contact",
            "policy",
            "how long",
            "what are",
            "what is",
        ]

        if any(
            keyword in text
            for keyword in rag_keywords
        ):

            return RouterDecision(
                action="rag",
                tool_name=None,
                args={},
            )

        # -----------------------------------------------------
        # Unknown request
        # -----------------------------------------------------

        return RouterDecision(
            action="escalate",
            tool_name=None,
            args={},
        )

    # =========================================================
    # OPENAI ROUTER
    # =========================================================

    effective_api_key = (
        api_key
        or os.getenv("OPENAI_API_KEY")
    )

    if client is None:

        if (
            not effective_api_key
            or not effective_api_key.strip()
        ):
            return RouterDecision(
                action="escalate",
                tool_name=None,
                args={},
            )

        from openai import OpenAI

        client = OpenAI(
            api_key=effective_api_key
        )

    messages = [
        {
            "role": "system",
            "content": ROUTER_SYSTEM_PROMPT,
        }
    ]

    if history:

        for msg in history[-4:]:

            messages.append(
                {
                    "role": msg.get(
                        "role",
                        "user",
                    ),
                    "content": msg.get(
                        "content",
                        "",
                    ),
                }
            )

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    response = client.chat.completions.create(
        model=os.getenv(
            "OPENAI_MODEL",
            model,
        ),
        messages=messages,
        response_format={
            "type": "json_object"
        },
        temperature=0.0,
    )

    raw_content = (
        response
        .choices[0]
        .message
        .content
    )

    data = json.loads(
        raw_content
    )

    decision = RouterDecision(
        **data
    )

    if decision.action != "tool":

        decision.tool_name = None
        decision.args = {}

    else:

        if decision.tool_name not in (
            "check_order_status",
            "check_account_status",
        ):
            raise ValueError(
                f"Unauthorized tool requested: "
                f"{decision.tool_name}"
            )

    return decision
# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "RouterDecision",
    "route_query",
    "local_route_query",
]