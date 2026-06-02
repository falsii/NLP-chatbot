import json
import os
import random

import torch

from nltk_utils import tokenize, bag_of_words
from model import NeuralNet


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
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    return model, intents, all_words, tags, device


def get_response(user_message, model, intents, all_words, tags, device):
    sentence = tokenize(user_message)
    x = bag_of_words(sentence, all_words)

    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)

    with torch.no_grad():
        output = model(x)

    _, predicted = torch.max(output, dim=1)
    predicted_tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)
    confidence = probabilities[0][predicted.item()].item()

    confidence_threshold = 0.75

    if confidence < confidence_threshold:
        return "I am not sure about that. Could you please explain it in another way?"

    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "I don't know the answer to that yet."


def main():
    model, intents, all_words, tags, device = load_chatbot()

    print("Bot is ready! Type 'quit' to exit.")

    while True:
        user_message = input("You: ")

        if user_message.lower() in ["quit", "exit", "bye"]:
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