import re
from difflib import get_close_matches


def normalize_text(text):
    """
    Lowercase text, remove extra spaces, and keep simple normalized form.
    """

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)

    return text


def contains_any(text, keywords):
    """
    Check whether any keyword exists inside text.
    """

    return any(keyword in text for keyword in keywords)


def safety_rule_response(user_message):
    """
    Handles sensitive or unsafe user messages before the ML model.
    """

    message = normalize_text(user_message)

    medical_keywords = [
        "medical advice",
        "medicine",
        "doctor",
        "diagnose",
        "diagnosis",
        "chest pain",
        "illness",
        "disease",
        "symptoms",
        "treatment",
        "tablet",
        "drug",
        "painkiller"
    ]

    legal_keywords = [
        "legal advice",
        "sue",
        "court case",
        "lawyer",
        "file a case",
        "legal help",
        "legal issue",
        "lawsuit"
    ]

    financial_keywords = [
        "financial advice",
        "invest",
        "investment",
        "stock",
        "crypto",
        "trading",
        "mutual fund",
        "loan advice",
        "where should i invest"
    ]

    personal_data_keywords = [
        "personal information",
        "private data",
        "home address",
        "credit card",
        "bank account"
    ]

    dangerous_password_phrases = [
        "give me password",
        "show me password",
        "find password",
        "hack password",
        "steal password",
        "someone password",
        "someone's password",
        "get password",
        "share password"
    ]

    dangerous_email_phrases = [
        "give me email address",
        "show me email address",
        "find email address",
        "someone email address",
        "someone's email address",
        "private email address"
    ]

    offensive_keywords = [
        "hate speech",
        "offensive content",
        "hateful content",
        "abuse someone",
        "insult someone"
    ]

    abusive_keywords = [
        "stupid",
        "idiot",
        "shut up",
        "dumb",
        "useless",
        "trash",
        "rubbish",
        "nonsense",
        "worst bot",
        "i hate you"
    ]

    if contains_any(message, abusive_keywords):
        return {
            "handled": True,
            "response": "Please use respectful language so I can help you.",
            "source": "rule_safety",
            "intent": "abusive_language",
            "confidence": 1.0
        }

    if (
        contains_any(message, medical_keywords)
        or contains_any(message, legal_keywords)
        or contains_any(message, financial_keywords)
        or contains_any(message, personal_data_keywords)
        or contains_any(message, dangerous_password_phrases)
        or contains_any(message, dangerous_email_phrases)
        or contains_any(message, offensive_keywords)
    ):
        return {
            "handled": True,
            "response": "I am sorry, but I cannot help with medical, legal, financial, offensive, or personal data-related requests. Please contact a qualified professional if needed.",
            "source": "rule_safety",
            "intent": "sensitive_content",
            "confidence": 1.0
        }

    return {
        "handled": False
    }


def shortcut_rule_response(user_message):
    """
    Handles very common short messages before ML model.
    This improves accuracy for small inputs like hi, bye, price, help, ok.
    """

    message = normalize_text(user_message)

    direct_responses = {
        "hi": {
            "response": "Hello! How can I help you today?",
            "intent": "greetings"
        },
        "hello": {
            "response": "Hello! How can I help you today?",
            "intent": "greetings"
        },
        "hey": {
            "response": "Hey! I am here to help you.",
            "intent": "greetings"
        },
        "bye": {
            "response": "Goodbye! Have a great day.",
            "intent": "goodbye"
        },
        "goodbye": {
            "response": "Goodbye! Have a great day.",
            "intent": "goodbye"
        },
        "thanks": {
            "response": "You're welcome!",
            "intent": "thanks"
        },
        "thank you": {
            "response": "You're welcome!",
            "intent": "thanks"
        },
        "price": {
            "response": "You can check the pricing section to see available plans and costs.",
            "intent": "pricing"
        },
        "pricing": {
            "response": "You can check the pricing section to see available plans and costs.",
            "intent": "pricing"
        },
        "cost": {
            "response": "You can check the pricing section to see available plans and costs.",
            "intent": "pricing"
        },
        "help": {
            "response": "Sure, I can help. Please tell me what issue you are facing.",
            "intent": "support"
        },
        "support": {
            "response": "Sure, I can help. Please tell me what issue you are facing.",
            "intent": "support"
        },
        "ok": {
            "response": "Okay.",
            "intent": "confirmation"
        },
        "okay": {
            "response": "Okay.",
            "intent": "confirmation"
        },
        "yes": {
            "response": "Got it.",
            "intent": "confirmation"
        },
        "no": {
            "response": "Okay, no problem.",
            "intent": "negative_confirmation"
        }
    }

    if message in direct_responses:
        item = direct_responses[message]

        return {
            "handled": True,
            "response": item["response"],
            "source": "rule_shortcut",
            "intent": item["intent"],
            "confidence": 1.0
        }

    return {
        "handled": False
    }


def identity_rule_response(user_message):
    """
    Handles common bot identity questions.
    This prevents confusion with abusive_language intent.
    """

    message = normalize_text(user_message)

    identity_patterns = [
        "are you human",
        "are you a human",
        "are you real",
        "are you a real person",
        "are you a bot",
        "are you chatbot",
        "are you a chatbot",
        "are you an ai",
        "are you robot",
        "are you a robot",
        "who are you",
        "what are you",
        "tell me about yourself",
        "am i talking to a human",
        "am i talking to a bot"
    ]

    if message in identity_patterns:
        return {
            "handled": True,
            "response": "I am a chatbot, not a human. I am here to help you.",
            "source": "rule_identity",
            "intent": "bot_identity",
            "confidence": 1.0
        }

    return {
        "handled": False
    }


def fuzzy_shortcut_intent(user_message):
    """
    Handles small spelling mistakes for short one-word messages.

    Example:
    helo -> hello
    prcie -> price
    suport -> support
    thnaks -> thanks
    """

    message = normalize_text(user_message)

    # Only apply fuzzy matching to short messages.
    # Avoid fuzzy matching long sentences because it may create wrong matches.
    if len(message.split()) > 2:
        return {
            "handled": False
        }

    keyword_to_response = {
        "hello": {
            "response": "Hello! How can I help you today?",
            "intent": "greetings"
        },
        "hi": {
            "response": "Hello! How can I help you today?",
            "intent": "greetings"
        },
        "bye": {
            "response": "Goodbye! Have a great day.",
            "intent": "goodbye"
        },
        "thanks": {
            "response": "You're welcome!",
            "intent": "thanks"
        },
        "price": {
            "response": "You can check the pricing section to see available plans and costs.",
            "intent": "pricing"
        },
        "pricing": {
            "response": "You can check the pricing section to see available plans and costs.",
            "intent": "pricing"
        },
        "support": {
            "response": "Sure, I can help. Please tell me what issue you are facing.",
            "intent": "support"
        },
        "help": {
            "response": "Sure, I can help. Please tell me what issue you are facing.",
            "intent": "support"
        }
    }

    matches = get_close_matches(
        message,
        keyword_to_response.keys(),
        n=1,
        cutoff=0.8
    )

    if matches:
        matched_keyword = matches[0]
        item = keyword_to_response[matched_keyword]

        return {
            "handled": True,
            "response": item["response"],
            "source": "rule_fuzzy_shortcut",
            "intent": item["intent"],
            "confidence": 0.9
        }

    return {
        "handled": False
    }


def apply_rules(user_message):
    """
    Main rule engine.

    Priority:
    1. Safety rules
    2. Bot identity rules
    3. Direct shortcut rules
    4. Fuzzy shortcut rules
    5. If no rule matches, allow ML model
    """

    rule_checks = [
        safety_rule_response,
        identity_rule_response,
        shortcut_rule_response,
        fuzzy_shortcut_intent
    ]

    for rule in rule_checks:
        result = rule(user_message)

        if result.get("handled"):
            return result

    return {
        "handled": False
    }