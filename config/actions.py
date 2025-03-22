from typing import Optional

from nemoguardrails.actions import action

@action(is_system_action=True)
async def check_blocked_terms(context: Optional[dict] = None):
    bot_response = context.get("bot_message")
    user_question=context.get("user_input")
    # A quick hard-coded list of offensive terms. You can also read this from a file.
    offensive_terms = [""]

    for term in offensive_terms:
        if term in bot_response.lower():
            return True

    return False

@action(is_system_action=True)
async def check_blocked_terms_user(context: Optional[dict] = None):
    bot_response = context.get("bot_message")
    user_question=context.get("user_input")
    # A quick hard-coded list of offensive terms. You can also read this from a file.
    offensive_terms = [""]
    if user_question:
        for term in offensive_terms:
            if term in user_question.lower():
                return True

    return False