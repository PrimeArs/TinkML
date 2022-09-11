import argparse
import os
import pickle
from collections import defaultdict

import numpy as np


def defaultdict_gen():
    return defaultdict(int)


class Model:
    SEP = '.'

    def __init__(self, n):
        self.n = n
        self.count_n_grams = defaultdict(defaultdict_gen)
        self.count_words = defaultdict(int)

    def fit(self, text):
        words = self.preprocess(text)
        for word in words:
            if word is not self.SEP:
                self.count_words[word] += 1
        for i in range(len(words) - self.n):
            self.count_n_grams[tuple(words[i:i + self.n])][words[i + self.n]] += 1

    def preprocess(self, text: str):
        text = text.lower()
        for sentence_end in ('.', '...', '?', '!', '!!!', '???'):
            text.replace(sentence_end, self.SEP)
        text = ''.join([c for c in text if c.isalpha() or c.isspace() or c == self.SEP])
        text = text.replace(self.SEP, f' {self.SEP} ')
        words = [word for word in text.split() if not word.isspace()]
        return words

    def postprocess(self, words):
        text = ' '.join(words)
        text = text.replace(f' {self.SEP}', self.SEP)
        return text

    def reset(self):
        self.count_n_grams = defaultdict(lambda: defaultdict(int))

    def generate(self, words):
        words = tuple(words)
        if words not in self.count_n_grams:
            p = np.array(list(self.count_words.values())) / sum(self.count_words.values())
            return np.random.choice(list(self.count_words.keys()), p=p)
        p = np.array(list(self.count_n_grams[words].values())) / sum(self.count_n_grams[words].values())
        return np.random.choice(list(self.count_n_grams[words].keys()), p=p)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


def train(input_dir, model_path):
    model = Model(n=3)
    for file in os.listdir(input_dir):
        with open(os.path.join(input_dir, file), 'r', encoding='utf-8') as f:
            text = f.read()
            model.fit(text)

    model.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train model')
    parser.add_argument('--input-dir', dest='input_dir', type=str, help='Directory with texts')
    parser.add_argument('--model', dest='model_path', type=str, help='Path to file to save trained model')
    args = parser.parse_args()

    train(args.input_dir, args.model_path)
