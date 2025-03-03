"""
Code for Problem 1 of HW 1.
"""
from typing import Iterable

import numpy as np


class Embeddings:
    """
    Problem 1b: Complete the implementation of this class based on the
    docstrings and the usage examples in the problem set.

    This class represents a container that holds a collection of words
    and their corresponding word embeddings.
    """

    def __init__(self, words: Iterable[str], vectors: np.ndarray):
        """
        Initializes an Embeddings object directly from a list of words
        and their embeddings.

        :param words: A list of words
        :param vectors: A 2D array of shape (len(words), embedding_size)
            where for each i, vectors[i] is the embedding for words[i]
        """
        self.words = list(words) #list of words
        self.indices = {w: i for i, w in enumerate(words)} # indices to words
        self.vectors = vectors # intanciates the vectors

    def __len__(self): #length on instance of embedding
        return len(self.words)

    def __contains__(self, word: str) -> bool: # returns if word in list of words
        return word in self.words

    def __getitem__(self, words: Iterable[str]) -> np.ndarray:
        """
        Retrieves embeddings for a list of words.

        :param words: A list of words
        :return: A 2D array of shape (len(words), embedding_size) where
            for each i, the ith row is the embedding for words[i]

        """
        words = [words] if type(words) == str else words
        return np.array([self.vectors[self.indices[word]] for word in words])
    @classmethod
    def from_file(cls, filename: str) -> "Embeddings":
        """
        Initializes an Embeddings object from a .txt file containing
        word embeddings in GloVe format.

        :param filename: The name of the file containing the embeddings
        :return: An Embeddings object containing the loaded embeddings
        """
        words = []
        vectors = []

        with open(filename, "r") as file:
            for line in file:
                l = line.split()
                word = l[0]
                vector = np.array(l[1:], dtype= np.float32)
                words.append(word)
                vectors.append(vector)
        return cls(words, np.array(vectors))
