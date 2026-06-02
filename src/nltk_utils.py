import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import string

stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Split sentence into words/tokens.
    Example:
    "Hello, how are you?" -> ["Hello", ",", "how", "are", "you", "?"]
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Convert word to root form.
    Example:
    "pricing" -> "price"
    "running" -> "run"
    """
    return stemmer.stem(word.lower())


def clean_tokens(tokens):
    """
    Remove punctuation tokens.
    """
    return [token for token in tokens if token not in string.punctuation]


def bag_of_words(tokenized_sentence, all_words):
    """
    Convert sentence into bag-of-words vector.

    Example:
    sentence = ["hello", "how", "are", "you"]
    all_words = ["hi", "hello", "price", "support"]

    output = [0, 1, 0, 0]
    """

    tokenized_sentence = clean_tokens(tokenized_sentence)
    tokenized_sentence = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)

    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0

    return bag