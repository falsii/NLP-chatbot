import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FAQRetriever:
    def __init__(self, faq_path, similarity_threshold=0.35):
        self.faq_path = faq_path
        self.similarity_threshold = similarity_threshold

        self.faq_data = []
        self.questions = []
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = None

        self.load_faq_data()
        self.build_index()

    def load_faq_data(self):
        """
        Load FAQ data from local JSON file.
        """

        if not os.path.exists(self.faq_path):
            raise FileNotFoundError(f"FAQ file not found: {self.faq_path}")

        with open(self.faq_path, "r", encoding="utf-8") as file:
            self.faq_data = json.load(file)

        self.questions = [item["question"] for item in self.faq_data]

    def build_index(self):
        """
        Convert FAQ questions into TF-IDF vectors.
        """

        if not self.questions:
            raise ValueError("FAQ questions are empty.")

        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def search(self, user_message):
        """
        Search the most similar FAQ question for the user message.
        """

        user_vector = self.vectorizer.transform([user_message])

        similarities = cosine_similarity(user_vector, self.question_vectors)[0]

        best_index = similarities.argmax()
        best_score = similarities[best_index]

        best_faq = self.faq_data[best_index]

        if best_score < self.similarity_threshold:
            return {
                "matched": False,
                "score": float(best_score),
                "question": None,
                "answer": None,
                "category": None,
                "source": "faq_low_similarity"
            }

        return {
            "matched": True,
            "score": float(best_score),
            "question": best_faq["question"],
            "answer": best_faq["answer"],
            "category": best_faq.get("category"),
            "source": "faq_retrieval"
        }


def load_retriever(base_dir):
    faq_path = os.path.join(base_dir, "data", "faq.json")
    return FAQRetriever(faq_path=faq_path, similarity_threshold=0.35)