"""
Code for Problems 2 and 3 of HW 1.
"""
from typing import Dict, List, Tuple

import numpy as np

from embeddings import Embeddings

import re


def cosine_sim(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Problem 3b: Implement this function.

    Computes the cosine similarity between two matrices of row vectors.

    :param x: A 2D array of shape (m, embedding_size)
    :param y: A 2D array of shape (n, embedding_size)
    :return: An array of shape (m, n), where the entry in row i and
        column j is the cosine similarity between x[i] and y[j]
    """
    array = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        abs_x = np.linalg.norm(x[i])
        for j in range(len(y)):
            abs_y = np.linalg.norm(y[j])
            value = (x[i] @ y[j]) / (abs_x * abs_y)
            array[i,j] = value
    return array


def get_closest_words(embeddings: Embeddings, vectors: np.ndarray,
                      k: int = 1) -> List[List[str]]:
    """
    Problem 3c: Implement this function.

    Finds the top k words whose embeddings are closest to a given vector
    in terms of cosine similarity.

    :param embeddings: A set of word embeddings
    :param vectors: A 2D array of shape (m, embedding_size)
    :param k: The number of closest words to find for each vector
    :return: A list of m lists of words, where the ith list contains the
        k words that are closest to vectors[i] in the embedding space,
        not necessarily in order
    """
    closest_words = []
    for vector in vectors: # iterating over vector array
        vecs_list = [] # We need to store the top k words for each vector
        vector = vector.reshape(1,-1)# Shaping the vector a 2D array for correct computation
        for e in embeddings.words:
            cos_sim_val = cosine_sim(vector, embeddings[[e]])
            vecs_list.append((e, cos_sim_val))
        # Sorting for top k neighbors
        top_k = sorted(vecs_list,key=lambda x: x[1], reverse=True)[:k]
        get_closest_words = [word for word, _ in top_k]
        closest_words.append(get_closest_words)
    return closest_words


# This type alias represents the format that the testing data should be
# deserialized into. An analogy is a tuple of 4 strings, and an
# AnalogiesDataset is a dict that maps a relation type to the list of
# analogies under that relation type.
AnalogiesDataset = Dict[str, List[Tuple[str, str, str, str]]]


def load_analogies(filename: str) -> AnalogiesDataset:
    """
    Problem 2b: Implement this function.

    Loads testing data for 3CosAdd from a .txt file and deserializes it
    into the AnalogiesData format.

    :param filename: The name of the file containing the testing data
    :return: An AnalogiesDataset containing the data in the file. The
        format of the data is described in the problem set and in the
        docstring for the AnalogiesDataset type alias
    """
    data: AnalogiesDataset = {}
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(r":\s(.+)",line)
            if match:
                key = match.group(1)
                data[key] = []

            else:
                words = tuple(word.lower() for word in line.split())
                data[key].append(words)
    return data

def run_analogy_test(embeddings: Embeddings, test_data: AnalogiesDataset,
                     k: int = 1) -> Dict[str, float]:
    """
    Problem 3d: Implement this function.

    Runs the 3CosAdd test for a set of embeddings and analogy questions.

    :param embeddings: The embeddings to be evaluated using 3CosAdd
    :param test_data: The set of analogies with which to compute analogy
        question accuracy
    :param k: The "lenience" with which accuracy will be computed. An
        analogy question will be considered to have been answered
        correctly as long as the target word is within the top k closest
        words to the target vector, even if it is not the closest word
    :return: The results of the 3CosAdd test, in the format of a dict
        that maps each relation type to the analogy question accuracy
        attained by embeddings on analogies from that relation type
    """
    analogy_dict = {}

    # Iterating over test_data analogies
    for category, analogies in test_data.items():
        correct = 0
        total = 0
        for analogy in analogies:
            w1,w2,w3,w4 = analogy
            vecs = embeddings[[w1, w2, w3]]

            # Computing predicted vector
            predicted_vector = vecs[1:2] - vecs[0:1] + vecs[2:3]

            # Get closest words to the predicted vector
            closest_words = get_closest_words(embeddings, predicted_vector, k=k)

            # Check if predicted vector == w4
            if w4 in closest_words[0]:
                correct += 1
            total += 1

        # Compute accuracy for each cat
        accuracy = correct / total
        analogy_dict[category] = accuracy

    return analogy_dict
