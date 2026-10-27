"""Mock banking tools. Mock data only: nothing here touches a real account."""
import re

MOCK_CARDS = {"1234": "active", "5678": "active"}
MOCK_TRANSACTIONS = {
    "TXN1001": {"status": "completed", "amount": 25.00, "merchant": "Cafe Demo"},
    "TXN1002": {"status": "pending", "amount": 120.50, "merchant": "Online Store Demo"},
    "TXN1003": {"status": "failed", "amount": 60.00, "merchant": "ATM Demo"},
}


class ToolError(ValueError):
    """Invalid arguments or unknown record; message is safe to show to the user."""


def block_card(card_last4: str) -> dict:
    if not re.fullmatch(r"\d{4}", str(card_last4)):
        raise ToolError("card_last4 must be exactly 4 digits.")
    if card_last4 not in MOCK_CARDS:
        raise ToolError("No card ending in those digits was found.")
    MOCK_CARDS[card_last4] = "blocked"
    return {"card_last4": card_last4, "status": "blocked"}


def transaction_status(transaction_id: str) -> dict:
    if not re.fullmatch(r"TXN\d{4}", str(transaction_id)):
        raise ToolError("transaction_id must look like TXN1234.")
    if transaction_id not in MOCK_TRANSACTIONS:
        raise ToolError(f"Transaction {transaction_id} was not found.")
    return {"transaction_id": transaction_id, **MOCK_TRANSACTIONS[transaction_id]}


# name -> (function, required argument names)
TOOLS = {
    "block_card": (block_card, {"card_last4"}),
    "transaction_status": (transaction_status, {"transaction_id"}),
}


def run_tool(name: str, arguments: dict) -> dict:
    """Validate tool name and exact argument set, then run. Raises ToolError on any problem."""
    if name not in TOOLS:
        raise ToolError(f"Unknown tool '{name}'.")
    fn, required = TOOLS[name]
    if not isinstance(arguments, dict) or set(arguments) != required:
        raise ToolError(f"Tool '{name}' needs exactly these arguments: {sorted(required)}.")
    return fn(**arguments)


def describe_result(name: str, result: dict) -> str:
    """Final customer text, grounded in the tool result (demo only)."""
    if name == "block_card":
        return f"(Demo) Card ending {result['card_last4']} is now {result['status']}."
    return (f"(Demo) Transaction {result['transaction_id']} at {result['merchant']} "
            f"for {result['amount']:.2f} is {result['status']}.")
