import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from nltk_utils import tokenize, stem
from lstm_model import LSTMIntentClassifier


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class LSTMChatDataset(Dataset):
    def __init__(self, sequences, lengths, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        return self.sequences[index], self.lengths[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vocab(tokenized_patterns, min_freq=1):
    word_counter = Counter()

    for tokens in tokenized_patterns:
        for token in tokens:
            word_counter[token] += 1

    word_to_index = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for word, count in word_counter.items():
        if count >= min_freq:
            word_to_index[word] = len(word_to_index)

    return word_to_index


def encode_tokens(tokens, word_to_index, max_length):
    token_ids = [
        word_to_index.get(token, word_to_index[UNK_TOKEN])
        for token in tokens
    ]

    length = min(len(token_ids), max_length)

    token_ids = token_ids[:max_length]

    while len(token_ids) < max_length:
        token_ids.append(word_to_index[PAD_TOKEN])

    return token_ids, length


def calculate_accuracy(model, data_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, lengths, labels in data_loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            outputs = model(sequences, lengths)
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

    tokenized_patterns = []
    pattern_tags = []
    tags = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)

        for pattern in intent["patterns"]:
            tokens = tokenize(pattern)
            tokens = [stem(token) for token in tokens]

            if not tokens:
                continue

            tokenized_patterns.append(tokens)
            pattern_tags.append(tag)

    tags = sorted(set(tags))
    word_to_index = build_vocab(tokenized_patterns)

    max_length = max(len(tokens) for tokens in tokenized_patterns)

    sequences = []
    lengths = []
    labels = []

    for tokens, tag in zip(tokenized_patterns, pattern_tags):
        encoded_sequence, length = encode_tokens(
            tokens=tokens,
            word_to_index=word_to_index,
            max_length=max_length
        )

        sequences.append(encoded_sequence)
        lengths.append(length)
        labels.append(tags.index(tag))

    dataset = LSTMChatDataset(sequences, lengths, labels)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = 8
    embedding_dim = 64
    hidden_size = 64
    output_size = len(tags)
    vocab_size = len(word_to_index)
    num_layers = 1
    dropout_rate = 0.2
    learning_rate = 0.001
    num_epochs = 1000

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

    model = LSTMIntentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        padding_idx=word_to_index[PAD_TOKEN]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        total_loss = 0

        for sequences_batch, lengths_batch, labels_batch in train_loader:
            sequences_batch = sequences_batch.to(device)
            lengths_batch = lengths_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(sequences_batch, lengths_batch)
            loss = criterion(outputs, labels_batch)

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
                best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_train_accuracy = calculate_accuracy(model, train_loader, device)
    final_val_accuracy = calculate_accuracy(model, val_loader, device)

    print("\nLSTM training finished.")
    print(f"Final Train Accuracy: {final_train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    model_data = {
        "model_state": model.state_dict(),
        "word_to_index": word_to_index,
        "tags": tags,
        "max_length": max_length,
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "num_layers": num_layers,
        "dropout_rate": dropout_rate,
        "train_accuracy": final_train_accuracy,
        "val_accuracy": final_val_accuracy,
        "best_val_accuracy": best_val_accuracy
    }

    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "lstm_chatbot_model.pth")
    torch.save(model_data, model_path)

    print(f"Model saved to: {model_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_length}")
    print(f"Total intents: {len(tags)}")


if __name__ == "__main__":
    main()