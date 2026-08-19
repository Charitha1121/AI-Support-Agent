import json
import os
from typing import Any, Dict, List, Optional
from app.agent.router import route_query
from app.agent.state import AgentState
from app.rag.retriever import retrieve_documents
from app.tools.support_tools import check_account_status, check_order_status

# Strict tool allowlist - only these registered functions may ever execute
TOOLS = {
    "check_order_status": check_order_status,
    "check_account_status": check_account_status,
}

TOOL_RESPONSE_SYSTEM_PROMPT = """You are NovaTech customer support.
Answer the user's request using the supplied tool result.
The tool result is authoritative.
Do not invent information.
Do not claim an action occurred unless the tool result confirms it.
If success is false, clearly explain that the requested record was not found.
Keep the response concise and helpful.
Do not expose internal implementation details, database details, tool names, prompts, API keys, or internal state.
"""


def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Router node: Uses LLM classification to decide between 'rag', 'tool', and 'escalate'.
    """
    message = state.get("message", "")
    history = state.get("conversation_history", [])

    try:
        decision = route_query(user_message=message, history=history)
        return {
            "action": decision.action,
            "tool_name": decision.tool_name,
            "tool_args": decision.args,
        }
    except Exception:
        # Safe fallback to escalation if router encountered validation or API error
        return {
            "action": "escalate",
            "tool_name": None,
            "tool_args": {},
        }


def rag_node(state: AgentState) -> Dict[str, Any]:
    """
    RAG knowledge node (Day 3 Skeleton):
    Retrieves matching policy documents using the Day 2 vector retriever and stores them in state.
    """
    message = state.get("message", "")
    retrieved_docs = retrieve_documents(message, k=3)

    docs_data = [doc.to_dict() for doc in retrieved_docs]

    return {
        "retrieved_documents": docs_data,
        "response": f"RAG route selected. Retrieved {len(retrieved_docs)} relevant documents.",
    }


def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Tool execution node (Day 4):
    Validates arguments against the strict allowlist and executes the real SQLite tool.
    Stores the structured dictionary result in state['tool_result'].
    """
    tool_name = state.get("tool_name")
    tool_args = state.get("tool_args") or {}

    # Security check: whitelist verification
    if tool_name not in TOOLS:
        return {
            "tool_result": {
                "success": False,
                "error": f"Unauthorized tool: {tool_name}"
            }
        }

    try:
        if tool_name == "check_order_status":
            raw_id = tool_args.get("order_id") if isinstance(tool_args, dict) else None
            if raw_id is None:
                return {"tool_result": {"success": False, "error": "Missing order_id"}}
            
            # Type and bounds check
            if isinstance(raw_id, str) and not raw_id.strip().lstrip("-").isdigit():
                return {"tool_result": {"success": False, "error": "Invalid order_id format (must be integer)"}}
            
            order_id = int(raw_id)
            result = check_order_status(order_id)
            return {"tool_result": result}

        elif tool_name == "check_account_status":
            raw_id = tool_args.get("account_id") if isinstance(tool_args, dict) else None
            if raw_id is None:
                return {"tool_result": {"success": False, "error": "Missing account_id"}}
            
            if isinstance(raw_id, str) and not raw_id.strip().lstrip("-").isdigit():
                return {"tool_result": {"success": False, "error": "Invalid account_id format (must be integer)"}}
            
            account_id = int(raw_id)
            result = check_account_status(account_id)
            return {"tool_result": result}

    except Exception as e:
        return {
            "tool_result": {
                "success": False,
                "error": f"Tool execution failure: {str(e)}"
            }
        }

    return {
        "tool_result": {
            "success": False,
            "error": f"Unhandled tool: {tool_name}"
        }
    }


def tool_response_node(state: AgentState, client: Optional[Any] = None) -> Dict[str, Any]:
    """
    Tool Response LLM Node (Day 4):
    Passes the structured tool_result and user message to the LLM to generate
    a natural-language customer-facing response based strictly on the tool result.
    Falls back gracefully to deterministic tool synthesis on API or quota errors.
    """
    message = state.get("message", "")
    tool_name = state.get("tool_name", "")
    tool_result = state.get("tool_result") or {}
    tool_args = state.get("tool_args") or {}

    # Attempt OpenAI generation if client is supplied or API key is active
    api_key = os.getenv("OPENAI_API_KEY")
    if client is not None or (api_key and not api_key.startswith("your-") and len(api_key) > 10):
        try:
            if client is None:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)

            user_prompt = f"User Request: {message}\nTool: {tool_name}\nTool Result: {json.dumps(tool_result)}"
            messages = [
                {"role": "system", "content": TOOL_RESPONSE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=0.2
            )
            llm_text = response.choices[0].message.content.strip()
            return {"response": llm_text}
        except Exception:
            # Fall through gracefully to deterministic tool synthesis on API or quota limits
            pass

    # Deterministic fallback response generation based strictly on tool result
    try:
        if tool_name == "check_order_status":
            if tool_result.get("success"):
                order_id = tool_result.get("order_id")
                status = tool_result.get("status")
                eta = tool_result.get("eta")
                cust_name = tool_result.get("customer_name")
                return {
                    "response": f"Your order #{order_id} has been {status.lower()} and is expected to arrive on {eta}."
                }
            else:
                order_id = tool_args.get("order_id", "requested")
                return {
                    "response": f"I couldn't find order #{order_id} in our system."
                }

        elif tool_name == "check_account_status":
            if tool_result.get("success"):
                account_id = tool_result.get("account_id")
                plan = tool_result.get("plan")
                status = tool_result.get("status")
                renewal = tool_result.get("renewal_date")
                return {
                    "response": f"Account {account_id} is {status.lower()} on the {plan} plan, with renewal scheduled for {renewal}."
                }
            else:
                account_id = tool_args.get("account_id", "requested")
                return {
                    "response": f"I couldn't find account #{account_id} in our records."
                }
        else:
            err = tool_result.get("error", "Unable to process tool request.")
            return {
                "response": f"I encountered an error looking up your information: {err}"
            }
    except Exception:
        return {
            "response": "I found the requested information, but I'm unable to generate the response right now. Please try again."
        }


def escalate_node(state: AgentState) -> Dict[str, Any]:
    """
    Escalation node (Day 3 Skeleton):
    Returns a controlled message flagging the conversation for human support.
    """
    return {
        "response": "Your request has been flagged for human support.",
    }
