import json
import os
import random

import torch

from nltk_utils import tokenize, stem
from lstm_model import LSTMIntentClassifier
from rules import apply_rules
from retrieval import load_retriever


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
DEBUG_MODE = True


def load_lstm_chatbot():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    intents_path = os.path.join(base_dir, "data", "intents.json")
    model_path = os.path.join(base_dir, "model", "lstm_chatbot_model.pth")

    with open(intents_path, "r", encoding="utf-8") as file:
        intents = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(model_path, map_location=device)

    word_to_index = data["word_to_index"]
    tags = data["tags"]
    max_length = data["max_length"]

    model = LSTMIntentClassifier(
        vocab_size=data["vocab_size"],
        embedding_dim=data["embedding_dim"],
        hidden_size=data["hidden_size"],
        output_size=data["output_size"],
        num_layers=data["num_layers"],
        dropout_rate=data["dropout_rate"],
        padding_idx=word_to_index[PAD_TOKEN]
    ).to(device)

    model.load_state_dict(data["model_state"])
    model.eval()

    retriever = load_retriever(base_dir)

    return model, intents, word_to_index, tags, max_length, device, retriever


def encode_message(user_message, word_to_index, max_length):
    tokens = tokenize(user_message)
    tokens = [stem(token) for token in tokens]

    if not tokens:
        tokens = [UNK_TOKEN]

    token_ids = [
        word_to_index.get(token, word_to_index[UNK_TOKEN])
        for token in tokens
    ]

    length = min(len(token_ids), max_length)

    token_ids = token_ids[:max_length]

    while len(token_ids) < max_length:
        token_ids.append(word_to_index[PAD_TOKEN])

    sequence = torch.tensor([token_ids], dtype=torch.long)
    length_tensor = torch.tensor([length], dtype=torch.long)

    return sequence, length_tensor


def get_intent_response(predicted_tag, intents):
    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "I don't know the answer to that yet."


def get_lstm_prediction(user_message, model, word_to_index, tags, max_length, device):
    sequence, length = encode_message(
        user_message=user_message,
        word_to_index=word_to_index,
        max_length=max_length
    )

    sequence = sequence.to(device)
    length = length.to(device)

    with torch.no_grad():
        output = model(sequence, length)

    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, dim=1)

    predicted_tag = tags[predicted.item()]
    confidence_score = confidence.item()

    return predicted_tag, confidence_score


def should_use_faq_retrieval(user_message, predicted_tag=None):
    message = user_message.lower().strip()

    faq_question_words = [
        "how",
        "what",
        "where",
        "when",
        "why",
        "can",
        "do",
        "does",
        "is",
        "are"
    ]

    faq_related_keywords = [
        "account",
        "password",
        "email",
        "delete",
        "price",
        "pricing",
        "cost",
        "plan",
        "subscription",
        "billing",
        "payment",
        "feature",
        "support",
        "bug",
        "issue",
        "data",
        "security"
    ]

    non_faq_intents = [
        "greetings",
        "goodbye",
        "thanks",
        "confirmation",
        "negative_confirmation",
        "small_talk",
        "bot_identity",
        "abusive_language",
        "sensitive_content"
    ]

    if predicted_tag in non_faq_intents:
        return False

    if any(message.startswith(word + " ") for word in faq_question_words):
        return True

    if any(keyword in message for keyword in faq_related_keywords):
        return True

    return False


def predict_lstm_chatbot(
    user_message,
    model,
    intents,
    word_to_index,
    tags,
    max_length,
    device,
    retriever
):
    rule_result = apply_rules(user_message)

    if rule_result.get("handled"):
        return {
            "message": user_message,
            "response": rule_result["response"],
            "intent": rule_result["intent"],
            "confidence": rule_result["confidence"],
            "source": rule_result["source"],
            "faq_match_score": None,
            "faq_question": None,
            "model_type": "lstm"
        }

    predicted_tag, confidence = get_lstm_prediction(
        user_message=user_message,
        model=model,
        word_to_index=word_to_index,
        tags=tags,
        max_length=max_length,
        device=device
    )

    if should_use_faq_retrieval(user_message, predicted_tag):
        faq_result = retriever.search(user_message)

        if faq_result["matched"]:
            return {
                "message": user_message,
                "response": faq_result["answer"],
                "intent": predicted_tag,
                "confidence": confidence,
                "source": "faq_retrieval",
                "faq_match_score": faq_result["score"],
                "faq_question": faq_result["question"],
                "model_type": "lstm"
            }

    confidence_threshold = 0.85

    if confidence < confidence_threshold:
        return {
            "message": user_message,
            "response": "I am not sure about that. Could you please explain it in another way?",
            "intent": "unknown",
            "confidence": confidence,
            "source": "model_low_confidence",
            "faq_match_score": None,
            "faq_question": None,
            "model_type": "lstm"
        }

    response = get_intent_response(predicted_tag, intents)

    return {
        "message": user_message,
        "response": response,
        "intent": predicted_tag,
        "confidence": confidence,
        "source": "model",
        "faq_match_score": None,
        "faq_question": None,
        "model_type": "lstm"
    }


def main():
    model, intents, word_to_index, tags, max_length, device, retriever = load_lstm_chatbot()

    print("LSTM Bot is ready! Type 'quit' to exit.")

    while True:
        user_message = input("You: ")

        if user_message.lower().strip() in ["quit", "exit"]:
            print("Bot: Goodbye! Have a great day.")
            break

        result = predict_lstm_chatbot(
            user_message=user_message,
            model=model,
            intents=intents,
            word_to_index=word_to_index,
            tags=tags,
            max_length=max_length,
            device=device,
            retriever=retriever
        )

        if DEBUG_MODE:
            debug_text = (
                f"DEBUG: model={result['model_type']} "
                f"source={result['source']} "
                f"intent={result['intent']} "
                f"confidence={result['confidence']:.2f}"
            )

            if result.get("faq_match_score") is not None:
                debug_text += f" faq_score={result['faq_match_score']:.2f}"

            if result.get("faq_question"):
                debug_text += f" faq_question='{result['faq_question']}'"

            print(debug_text)

        print(f"Bot: {result['response']}")


if __name__ == "__main__":
    main()