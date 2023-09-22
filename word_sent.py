import re
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm


class WordToSentVector:
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.sentence_vectors = None

    def tokenize(self, text):
        text = re.sub(r'[\n+]', '', str(text))
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'(pagi|siang|sore)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(dok|dokter)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(terimakasih|terima kasih|alo|alodokter)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        text = word_tokenize(text)
        return text

    def word_to_sent(self):
        sentence_vectors = np.array([
            np.mean([self.model[word] for word in words if word in self.model], axis=0)
            if any(word in self.model for word in words)
            else np.zeros(self.model.vector_size)
            for words in tqdm([self.tokenize(sentence) for sentence in self.data], leave=True)
        ])

        return sentence_vectors

    def fit(self):
        self.sentence_vectors = self.word_to_sent()
        return self.sentence_vectors
