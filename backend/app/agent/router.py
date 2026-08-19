import json
import os
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class RouterDecision(BaseModel):
    """
    Structured routing decision schema for NovaTech customer support.
    """
    action: Literal["rag", "tool", "escalate"] = Field(
        ...,
        description="Must be 'rag' for policy/knowledge base inquiries, 'tool' for order/account lookups, or 'escalate' for human support/unsupported questions."
    )
    tool_name: Optional[Literal["check_order_status", "check_account_status"]] = Field(
        default=None,
        description="Specific tool name if action is 'tool', otherwise null."
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the selected tool (e.g. {'order_id': 4521} or {'account_id': 1001})."
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        if v not in ("rag", "tool", "escalate"):
            raise ValueError(f"Invalid action '{v}'. Allowed actions are: 'rag', 'tool', 'escalate'.")
        return v

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v: Optional[str], info) -> Optional[str]:
        if v is not None and v not in ("check_order_status", "check_account_status"):
            raise ValueError(f"Invalid tool_name '{v}'. Allowed tools are: 'check_order_status', 'check_account_status'.")
        return v


ROUTER_SYSTEM_PROMPT = """You are the intelligent query classification router for NovaTech Customer Support.
Analyze the user's message and output a JSON decision conforming to the schema.

CATEGORIES:
1. "rag":
   Use for general company policy, FAQ, subscription tiers, pricing, shipping delivery timelines, return windows, password security, cancellation policy, payment methods, support contact hours.
   Example: "What is your refund policy?", "How long does shipping take?", "What subscription plans do you offer?"
   Output: { "action": "rag", "tool_name": null, "args": {} }

2. "tool":
   Use when the user is inquiring about live/account/order-specific details by providing an identifier.
   Allowed tools:
   - "check_order_status": for order status / tracking inquiries with an order ID.
     Example: "Where is order 4521?", "What's the status of order 4522?"
     Output: { "action": "tool", "tool_name": "check_order_status", "args": { "order_id": 4521 } }
   - "check_account_status": for account status / renewal / plan inquiries with an account ID.
     Example: "Is account 1001 active?", "What plan is account 1001 on?"
     Output: { "action": "tool", "tool_name": "check_account_status", "args": { "account_id": 1001 } }

3. "escalate":
   Use when:
   - The user explicitly asks to speak with a human / representative / supervisor.
   - The issue involves unresolved billing disputes or serious complaints.
   - The request is completely outside the scope of NovaTech support (e.g. legal advice, coding, unrelated general trivia).
   - The user asks for specific order/account info but provides no ID and cannot be answered by FAQ.
   Output: { "action": "escalate", "tool_name": null, "args": {} }

OUTPUT FORMAT: Return only a valid JSON object matching the schema above.
"""


def route_query(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    client: Optional[Any] = None
) -> RouterDecision:
    """
    Classify the user's query into 'rag', 'tool', or 'escalate' using OpenAI Structured Outputs.
    Validates output strictly with Pydantic and safe allowlisting.
    """
    effective_api_key = api_key or os.getenv("OPENAI_API_KEY")

    # Use provided client (for testing/mocking) or initialize OpenAI client
    if client is None:
        if not effective_api_key or not effective_api_key.strip():
            # In testing without API key, safe fallback to escalate or validation
            return RouterDecision(action="escalate", tool_name=None, args={})
        from openai import OpenAI
        client = OpenAI(api_key=effective_api_key)

    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
    if history:
        for msg in history[-4:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", model),
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0
    )

    raw_content = response.choices[0].message.content
    data = json.loads(raw_content)

    # Validate with Pydantic
    decision = RouterDecision(**data)

    # Security check: if action != 'tool', ensure tool_name is None and args are empty
    if decision.action != "tool":
        decision.tool_name = None
        decision.args = {}
    else:
        if decision.tool_name not in ("check_order_status", "check_account_status"):
            raise ValueError(f"Unauthorized tool requested: {decision.tool_name}")

    return decision
