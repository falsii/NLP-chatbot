import json
import os
import random

import torch

from nltk_utils import tokenize, bag_of_words
from model import NeuralNet
from rules import apply_rules


DEBUG_MODE = True


def load_chatbot():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    intents_path = os.path.join(base_dir, "data", "intents.json")
    model_path = os.path.join(base_dir, "model", "chatbot_model.pth")

    with open(intents_path, "r", encoding="utf-8") as file:
        intents = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(model_path, map_location=device)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    dropout_rate = data.get("dropout_rate", 0.2)

    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout_rate=dropout_rate
    ).to(device)

    model.load_state_dict(model_state)
    model.eval()

    return model, intents, all_words, tags, device


def get_intent_response(predicted_tag, intents):
    """
    Return random response for predicted intent.
    """

    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "I don't know the answer to that yet."


def get_model_prediction(user_message, model, all_words, tags, device):
    """
    Predict intent using PyTorch model.
    """

    sentence = tokenize(user_message)
    x = bag_of_words(sentence, all_words)

    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)

    with torch.no_grad():
        output = model(x)

    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, dim=1)

    predicted_tag = tags[predicted.item()]
    confidence_score = confidence.item()

    return predicted_tag, confidence_score


def get_response(user_message, model, intents, all_words, tags, device):
    """
    Main chatbot response function.

    Step 1: Apply rule-based layer.
    Step 2: If no rule matches, use PyTorch model.
    Step 3: If confidence is low, return unknown response.
    """

    rule_result = apply_rules(user_message)

    if rule_result.get("handled"):
        if DEBUG_MODE:
            print(
                f"DEBUG: source={rule_result['source']} "
                f"intent={rule_result['intent']} "
                f"confidence={rule_result['confidence']:.2f}"
            )

        return rule_result["response"]

    predicted_tag, confidence = get_model_prediction(
        user_message=user_message,
        model=model,
        all_words=all_words,
        tags=tags,
        device=device
    )

    if DEBUG_MODE:
        print(
            f"DEBUG: source=model "
            f"intent={predicted_tag} "
            f"confidence={confidence:.2f}"
        )

    confidence_threshold = 0.85

    if confidence < confidence_threshold:
        return "I am not sure about that. Could you please explain it in another way?"

    return get_intent_response(predicted_tag, intents)


def main():
    model, intents, all_words, tags, device = load_chatbot()

    print("Bot is ready! Type 'quit' to exit.")

    while True:
        user_message = input("You: ")

        if user_message.lower().strip() in ["quit", "exit"]:
            print("Bot: Goodbye! Have a great day.")
            break

        response = get_response(
            user_message=user_message,
            model=model,
            intents=intents,
            all_words=all_words,
            tags=tags,
            device=device
        )

        print(f"Bot: {response}")


if __name__ == "__main__":
    main()