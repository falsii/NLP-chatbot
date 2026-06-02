import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet


class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)
        self.n_samples = len(x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def set_seed(seed=42):
    """
    Make training results more consistent.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_accuracy(model, data_loader, device):
    """
    Calculate model accuracy.
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for words, labels in data_loader:
            words = words.to(device)
            labels = labels.to(device)

            outputs = model(words)
            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()

    if total == 0:
        return 0

    return correct / total


def main():
    set_seed(42)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    intents_path = os.path.join(base_dir, "data", "intents.json")

    with open(intents_path, "r", encoding="utf-8") as file:
        intents = json.load(file)

    all_words = []
    tags = []
    xy = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)

        for pattern in intent["patterns"]:
            tokens = tokenize(pattern)
            all_words.extend(tokens)
            xy.append((tokens, tag))

    all_words = [stem(word) for word in all_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    x_data = []
    y_data = []

    for pattern_tokens, tag in xy:
        bow = bag_of_words(pattern_tokens, all_words)
        x_data.append(bow)

        label = tags.index(tag)
        y_data.append(label)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    dataset = ChatDataset(x_data, y_data)

    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Hyperparameters
    input_size = len(x_data[0])
    hidden_size = 64
    output_size = len(tags)
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 1000
    dropout_rate = 0.2

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        total_loss = 0

        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            train_accuracy = calculate_accuracy(model, train_loader, device)
            val_accuracy = calculate_accuracy(model, val_loader, device)

            avg_loss = total_loss / len(train_loader)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Loss: {avg_loss:.4f} "
                f"Train Acc: {train_accuracy:.4f} "
                f"Val Acc: {val_accuracy:.4f}"
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

    train_accuracy = calculate_accuracy(model, train_loader, device)
    val_accuracy = calculate_accuracy(model, val_loader, device)

    print("\nTraining finished.")
    print(f"Final Train Accuracy: {train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "dropout_rate": dropout_rate,
        "all_words": all_words,
        "tags": tags,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "best_val_accuracy": best_val_accuracy
    }

    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "chatbot_model.pth")
    torch.save(data, model_path)

    print(f"Model saved to: {model_path}")
    print(f"Total patterns: {len(x_data)}")
    print(f"Vocabulary size: {len(all_words)}")
    print(f"Total intents: {len(tags)}")


if __name__ == "__main__":
    main()