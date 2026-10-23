"""Parse and validate the assistant's machine-readable response."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from src.safety.guardrails import SAFE_FALLBACK, check_output

INTENTS = frozenset("""activate_my_card age_limit apple_pay_or_google_pay atm_support automatic_top_up
balance_not_updated_after_bank_transfer balance_not_updated_after_cheque_or_cash_deposit beneficiary_not_allowed
cash_withdrawal_charge cash_withdrawal_not_recognised cancel_transfer card_about_to_expire card_acceptance
card_arrival card_delivery_estimate card_linking card_not_working card_payment_fee_charged
card_payment_not_recognised card_payment_wrong_exchange_rate card_swallowed change_pin compromised_card
contactless_not_working country_support declined_card_payment declined_cash_withdrawal declined_transfer
direct_debit_payment_not_recognised disposable_card_limits edit_personal_details exchange_charge exchange_rate
exchange_via_app extra_charge_on_statement failed_transfer fiat_currency_support get_disposable_virtual_card
get_physical_card getting_spare_card getting_virtual_card lost_or_stolen_card lost_or_stolen_phone
order_physical_card passcode_forgotten pending_card_payment pending_cash_withdrawal pending_top_up pending_transfer
pin_blocked receiving_money Refund_not_showing_up request_refund reverted_card_payment? supported_cards_and_currencies
terminate_account top_up_by_bank_transfer_charge top_up_by_card_charge top_up_by_cash_or_cheque top_up_failed
top_up_limits top_up_reverted topping_up_by_card transaction_charged_twice transfer_fee_charged
transfer_into_account transfer_not_received_by_recipient transfer_timing unable_to_verify_identity
verify_my_identity verify_source_of_funds verify_top_up virtual_card_not_working visa_or_mastercard
why_verify_identity wrong_amount_of_cash_received wrong_exchange_rate_for_cash_withdrawal""".split())


@dataclass(frozen=True)
class AssistantOutput:
    intent: str
    response: str
    action_required: bool = False
    action: dict[str, Any] | None = None


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    value: AssistantOutput | None
    errors: tuple[str, ...] = ()


def output_schema() -> dict[str, Any]:
    """Return a JSON Schema (Draft 2020-12) matching the validated contract."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Banking77AssistantOutput", "type": "object", "additionalProperties": False,
        "required": ["intent", "response", "action_required", "action"],
        "properties": {
            "intent": {"type": "string", "enum": sorted(INTENTS)},
            "response": {"type": "string", "minLength": 1},
            "action_required": {"type": "boolean"},
            "action": {"type": ["object", "null"]},
        },
        "allOf": [{"if": {"properties": {"action_required": {"const": True}}},
                   "then": {"properties": {"action": {"type": "object"}}}},
                  {"if": {"properties": {"action_required": {"const": False}}},
                   "then": {"properties": {"action": {"type": "null"}}}}],
    }


def validate_output(value: Any) -> ValidationResult:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ValidationResult(False, None, ("output must be a JSON object",))
    required = {"intent", "response", "action_required", "action"}
    missing, extra = required - value.keys(), value.keys() - required
    if missing: errors.append(f"missing fields: {', '.join(sorted(missing))}")
    if extra: errors.append(f"unexpected fields: {', '.join(sorted(extra))}")
    intent, response = value.get("intent"), value.get("response")
    action_required, action = value.get("action_required"), value.get("action")
    if not isinstance(intent, str) or intent not in INTENTS: errors.append("intent is not a Banking77 label")
    if not isinstance(response, str) or not response.strip(): errors.append("response must be a non-empty string")
    elif (safety := check_output(response)).status != "allowed": errors.extend(safety.reasons)
    if not isinstance(action_required, bool): errors.append("action_required must be a boolean")
    if action_required is True and not isinstance(action, dict): errors.append("required action must be an object")
    if action_required is False and action is not None: errors.append("action must be null when action_required is false")
    if errors: return ValidationResult(False, None, tuple(errors))
    return ValidationResult(True, AssistantOutput(intent, response, action_required, action))


def parse_output(raw: str) -> ValidationResult:
    """Parse plain JSON or a single fenced JSON block; never repair model text."""
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    if fenced: text = fenced.group(1)
    try:
        return validate_output(json.loads(text))
    except (json.JSONDecodeError, TypeError) as exc:
        return ValidationResult(False, None, (f"invalid JSON: {exc}",))


def safe_fallback(intent: str | None = None) -> AssistantOutput:
    """Build a conservative fallback; an invalid model intent is discarded."""
    return AssistantOutput(intent if intent in INTENTS else "card_not_working", SAFE_FALLBACK)


def to_json(output: AssistantOutput) -> str:
    return json.dumps(asdict(output), ensure_ascii=False)
