import sys
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database.db import create_tables, seed_data
from app.schemas.chat import ChatRequest
from app.services.chat_service import process_chat_message


def run_evaluation_suite():
    create_tables()
    seed_data()

    test_cases = [
        {
            "id": 1,
            "description": "Company policy refund question",
            "message": "What is your refund policy?",
            "expected_action": "rag",
            "validation": lambda res: "refund" in res.response.lower() or "14" in res.response
        },
        {
            "id": 2,
            "description": "Shipping transit timeline question",
            "message": "How long does shipping take?",
            "expected_action": "rag",
            "validation": lambda res: "shipping" in res.response.lower() or "3 to 5" in res.response.lower()
        },
        {
            "id": 3,
            "description": "Specific live order lookup #4521",
            "message": "Where is order 4521?",
            "expected_action": "tool",
            "validation": lambda res: "4521" in res.response or "Rahul" in res.response or "Shipped" in res.response
        },
        {
            "id": 4,
            "description": "Specific live order lookup #4522",
            "message": "What is the status of order 4522?",
            "expected_action": "tool",
            "validation": lambda res: "4522" in res.response or "Priya" in res.response or "Processing" in res.response
        },
        {
            "id": 5,
            "description": "Specific account status lookup #1001",
            "message": "Is account 1001 active?",
            "expected_action": "tool",
            "validation": lambda res: "1001" in res.response or "Pro" in res.response or "Active" in res.response
        },
        {
            "id": 6,
            "description": "Account renewal query without identifier",
            "message": "When does my account renew?",
            "expected_action": "escalate",
            "validation": lambda res: res.escalation_id is not None or "ID" in res.response or "account" in res.response
        },
        {
            "id": 7,
            "description": "Explicit user request to talk with a human",
            "message": "I want to speak with a human.",
            "expected_action": "escalate",
            "validation": lambda res: res.escalation_id is not None and res.escalation_id.startswith("ESC-")
        },
        {
            "id": 8,
            "description": "Unrelated legal advice request",
            "message": "Can you solve my unrelated legal problem?",
            "expected_action": "escalate",
            "validation": lambda res: res.escalation_id is not None
        },
        {
            "id": 9,
            "description": "Multi-turn consecutive order status lookup",
            "multi_turn": [
                {"message": "Where is order 4521?", "expected_action": "tool"},
                {"message": "What about 4522?", "expected_action": "tool"}
            ]
        },
        {
            "id": 10,
            "description": "Completely unrelated prompt (baking recipe)",
            "message": "Tell me a recipe for baking chocolate cake with walnuts.",
            "expected_action": "escalate",
            "validation": lambda res: "cake" not in res.response.lower() or res.escalation_id is not None
        }
    ]

    print("=================================================================")
    print("AI SUPPORT AGENT — 10-CONVERSATION BENCHMARK EVALUATION")
    print("=================================================================\n")

    passed_count = 0
    total_count = len(test_cases)

    for case in test_cases:
        case_id = case["id"]
        desc = case["description"]

        if "multi_turn" in case:
            conv_id = f"eval-conv-multiturn-{case_id}"
            turn_success = True
            print(f"Test #{case_id}: {desc}")
            for turn_idx, turn in enumerate(case["multi_turn"], 1):
                req = ChatRequest(message=turn["message"], conversation_id=conv_id)
                res = process_chat_message(req)
                print(f"  Turn {turn_idx} -> User: '{turn['message']}' | Action Taken: '{res.action_taken}' (Expected: '{turn['expected_action']}')")
                if res.action_taken != turn["expected_action"]:
                    turn_success = False
            if turn_success:
                print("  => Result: PASSED\n")
                passed_count += 1
            else:
                print("  => Result: FAILED\n")

        else:
            conv_id = f"eval-conv-{case_id}"
            req = ChatRequest(message=case["message"], conversation_id=conv_id)
            res = process_chat_message(req)
            action_match = res.action_taken == case["expected_action"]
            content_match = case["validation"](res) if "validation" in case else True
            is_pass = action_match and content_match

            print(f"Test #{case_id}: {desc}")
            print(f"  User: '{case['message']}'")
            print(f"  Agent: {res.response[:80]}...")
            print(f"  Action Taken: '{res.action_taken}' (Expected: '{case['expected_action']}')")
            print(f"  => Result: {'PASSED' if is_pass else 'FAILED'}\n")
            if is_pass:
                passed_count += 1

    print("=================================================================")
    print(f"EVALUATION SUMMARY: {passed_count}/{total_count} Tests Passed ({passed_count/total_count*100:.1f}%)")
    print("=================================================================\n")
    assert passed_count == total_count, f"Evaluation benchmark failed: {passed_count}/{total_count}"


if __name__ == "__main__":
    run_evaluation_suite()
