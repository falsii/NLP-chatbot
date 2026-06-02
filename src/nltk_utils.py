import re
import string
import numpy as np
from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Convert sentence into clean word tokens.

    Example:
    "What's your pricing?" -> ["what", "s", "your", "pricing"]
    """

    sentence = sentence.lower()

    # Keep only words and numbers
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", sentence)

    return tokens


def stem(word):
    """
    Convert word to root form.

    Example:
    "pricing" -> "price"
    "running" -> "run"
    """

    return stemmer.stem(word.lower())


def normalize_sentence(sentence):
    """
    Clean and normalize user sentence.
    """

    sentence = sentence.lower().strip()

    # Remove punctuation
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence


def bag_of_words(tokenized_sentence, all_words):
    """
    Convert tokenized sentence into Bag of Words vector.

    Example:
    tokenized_sentence = ["hello", "price"]
    all_words = ["hello", "support", "price"]

    output = [1, 0, 1]
    """

    tokenized_sentence = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)

    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0

    return bag