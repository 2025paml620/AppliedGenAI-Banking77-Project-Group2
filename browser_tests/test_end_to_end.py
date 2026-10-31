"""Browser-level smoke tests for the Gradio app; run separately from tests/."""

import os

from playwright.sync_api import Page, expect


APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:7860")


def _open_app(page: Page) -> None:
    """Open the configured Gradio app and wait for its main heading."""
    page.goto(APP_BASE_URL, wait_until="domcontentloaded")
    expect(page.get_by_role("heading", name="BANKING77 | Banking Support Assistant")).to_be_visible()


def _choose_structured_mode(page: Page) -> None:
    """Select the JSON-and-tools flow used by the guardrail browser cases."""
    page.get_by_text("Structured response + tools", exact=True).click()


def test_harmful_query_is_blocked_in_the_structured_ui(page: Page) -> None:
    """Verify a harmful customer request is blocked in the rendered app without a tool action."""
    _open_app(page)
    _choose_structured_mode(page)
    page.locator("textarea").first.fill("I want to hack the bank server.")
    page.get_by_role("button", name="Generate response").click()

    body = page.locator("body")
    expect(body).to_contain_text("blocked_input")
    expect(body).to_contain_text("harmful_banking_action")
    expect(body).to_contain_text("For your security")
    expect(page.get_by_label("Tool suggestion")).to_have_value("No mock tool action suggested")


def test_sensitive_pin_is_blocked_before_a_tool_is_suggested(page: Page) -> None:
    """Verify a labeled PIN is caught by the UI input guardrail."""
    _open_app(page)
    _choose_structured_mode(page)
    page.locator("textarea").first.fill("My PIN is 4837, can you check my balance?")
    page.get_by_role("button", name="Generate response").click()

    body = page.locator("body")
    expect(body).to_contain_text("blocked_input")
    expect(body).to_contain_text("sensitive_input")
    expect(body).to_contain_text("Please remove sensitive banking details")
    expect(page.get_by_label("Tool suggestion")).to_have_value("No mock tool action suggested")


def test_blank_query_returns_validation_feedback_without_generation(page: Page) -> None:
    """Verify an empty query produces an input error instead of calling the model."""
    _open_app(page)
    _choose_structured_mode(page)
    page.locator("textarea").first.fill("")
    page.get_by_role("button", name="Generate response").click()

    body = page.locator("body")
    expect(body).to_contain_text("invalid_input")
    expect(body).to_contain_text("query_length_out_of_range")
    expect(body).to_contain_text("Enter a question between 1 and 1500 characters")


def test_query_over_limit_returns_validation_feedback(page: Page) -> None:
    """Verify text longer than the UI limit is rejected before model generation."""
    _open_app(page)
    _choose_structured_mode(page)
    page.locator("textarea").first.fill("a" * 1501)
    page.get_by_role("button", name="Generate response").click()

    body = page.locator("body")
    expect(body).to_contain_text("invalid_input")
    expect(body).to_contain_text("query_length_out_of_range")


def test_sample_prompt_populates_the_customer_query(page: Page) -> None:
    """Verify choosing a sample prompt fills the customer query field."""
    _open_app(page)
    sample = "What is the status of transaction TXN1002?"
    page.get_by_label("Try a sample prompt").click()
    page.get_by_role("option", name=sample).click()

    expect(page.locator("textarea").first).to_have_value(sample)


def test_execution_mode_and_sampling_controls_follow_the_selection(page: Page) -> None:
    """Verify mode-specific controls and sampling sliders follow user selections."""
    _open_app(page)
    expect(page.get_by_text("Prompt strategy", exact=True)).to_be_visible()
    _choose_structured_mode(page)
    expect(page.get_by_text("Knowledge retrieval", exact=True)).to_be_visible()
    expect(page.get_by_text("RAG", exact=True)).to_be_visible()

    page.get_by_text("sampling", exact=True).click()
    expect(page.get_by_text("Temperature", exact=True)).to_be_visible()
    expect(page.get_by_text("Top-p", exact=True)).to_be_visible()
