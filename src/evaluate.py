import json
import os
from collections import Counter, defaultdict

from chat import load_chatbot, predict_chatbot


def load_test_questions(base_dir):
    test_path = os.path.join(base_dir, "data", "test_questions.json")

    with open(test_path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    test_questions = load_test_questions(base_dir)

    model, intents, all_words, tags, device, retriever = load_chatbot()

    total = len(test_questions)
    correct = 0

    wrong_predictions = []
    source_counter = Counter()
    intent_counter = Counter()
    confusion_matrix = defaultdict(Counter)

    for item in test_questions:
        message = item["message"]
        expected_intent = item["expected_intent"]

        result = predict_chatbot(
            user_message=message,
            model=model,
            intents=intents,
            all_words=all_words,
            tags=tags,
            device=device,
            retriever=retriever
        )

        predicted_intent = result["intent"]
        confidence = result["confidence"]
        source = result["source"]

        source_counter[source] += 1
        intent_counter[predicted_intent] += 1
        confusion_matrix[expected_intent][predicted_intent] += 1

        if predicted_intent == expected_intent:
            correct += 1
        else:
            wrong_predictions.append(
                {
                    "message": message,
                    "expected": expected_intent,
                    "predicted": predicted_intent,
                    "confidence": confidence,
                    "source": source
                }
            )

    accuracy = correct / total if total > 0 else 0
    unknown_count = intent_counter["unknown"]
    unknown_rate = unknown_count / total if total > 0 else 0

    print("\n================ Evaluation Report ================")
    print(f"Total test questions: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Wrong predictions: {len(wrong_predictions)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Unknown rate: {unknown_rate:.4f}")

    print("\n================ Source Breakdown ================")
    for source, count in source_counter.items():
        percentage = count / total * 100
        print(f"{source}: {count} ({percentage:.2f}%)")

    print("\n================ Predicted Intent Breakdown ================")
    for intent, count in intent_counter.items():
        percentage = count / total * 100
        print(f"{intent}: {count} ({percentage:.2f}%)")

    print("\n================ Wrong Predictions ================")
    if not wrong_predictions:
        print("No wrong predictions. Great job!")
    else:
        for index, item in enumerate(wrong_predictions, start=1):
            print(f"\n{index}. Message: {item['message']}")
            print(f"   Expected: {item['expected']}")
            print(f"   Predicted: {item['predicted']}")
            print(f"   Confidence: {item['confidence']:.4f}")
            print(f"   Source: {item['source']}")

    print("\n================ Confusion Matrix ================")
    print("Format: Expected Intent -> Predicted Intent Counts\n")

    for expected_intent, predictions in confusion_matrix.items():
        print(f"{expected_intent}:")
        for predicted_intent, count in predictions.items():
            print(f"  {predicted_intent}: {count}")

    print("\n===================================================")


if __name__ == "__main__":
    evaluate()