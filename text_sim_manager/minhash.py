from typing import Tuple, List
from string import ascii_lowercase
import random
import re
from collections import Counter
from itertools import compress
import json
from pathlib import Path
from tqdm import tqdm
import time, datetime
import numpy as np
import pandas as pd
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from datasets import Dataset,DatasetDict
from nltk import tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer



def run():
    with open("/Users/samguercho/Projects/Paraphrasing/data/test/test.src", 'r') as f:
        texts = tokenize.sent_tokenize(f.read())
        for i in range(6):
            texts.extend(texts[:(len(texts)//2)])
        start = time.time()
        minhash = MinHash()
        sig, indexed_texts= minhash.fit_transform(texts)
        minhash.compare_hashes(sig, indexed_texts)
        timedelta = time.time() - start
        print(f'The total time to produce the df is {time.strftime("%H:%M:%S", time.gmtime(timedelta))} seconds')

class MinHash:
    """
    MinHash is a class ruling the MinHash Algorithm.
    Explaination of the algorithm in the following: https://en.wikipedia.org/wiki/MinHash
    This version is the LSH : https://en.wikipedia.org/wiki/Locality-sensitive_hashing

    2 Main funcitons to use:
    - compare_lsh_sigs: To compare LSH signatures between 2 texts
    - fit_predict: To create LSH pairs in a given corpus
    """
    def __init__(self,
                 n_hash_functions: int=20,
                 ngram_range=(1,3),
                 alphabet: str='en',
                 **kwargs):
        self.alphabet = self._get_alphabet(alphabet)
        vocab = [''.join(x) for x in itertools.product(self.alphabet, repeat=3)]
        self.cv = CountVectorizer(binary=True, ngram_range=ngram_range,
                                  analyzer='char',
                                  **kwargs,
                                  vocabulary=sorted(list(set(vocab))),
                                  lowercase=True)
        self.n_hashes = n_hash_functions
        self.max_token_value = 2 ** 16 - 1
        # To define: one prime number to use for permutation
        self.prime = 139_169
        self._a, self._b = self._get_permuters()
        self.table_pairs = np.zeros((1, self.n_hashes))

    def _get_alphabet(self, alphabet:str):
        if alphabet.lower() == "en":
            return 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVXYZ#: '
        elif alphabet.lower() == "ru":
            return "абвгдеёжзийклмнопрстуфхцчшщъыьэюя#: "
        elif alphabet.lower() == "he":
            return "#: אבגדהוזחטיכךלמםנןסעפףצץקרשת"

    def _get_permuters(self):
        """
        When having a different number of hash functions compared to necessary, creates new one and add it in the
        hash_permuters file
        """
        n_hashes = str(self.n_hashes)
        hash_parameters_path = Path('.').parent
        with open(hash_parameters_path / 'data' / 'hash_parameters.json', 'r') as f_in:
            hash_permuters_json = json.load(f_in)
        if n_hashes not in hash_permuters_json.keys():
            a = np.random.randint(0, self.max_token_value, self.n_hashes)
            b = np.random.randint(0, self.max_token_value, self.n_hashes)
            hash_permuters_json[self.n_hashes] = dict()
            hash_permuters_json[self.n_hashes]['a'] = a.tolist()
            hash_permuters_json[self.n_hashes]['b'] = b.tolist()
            with open(hash_parameters_path / 'data' / 'hash_parameters.json', 'w') as f_out:
                json.dump(hash_permuters_json, f_out)

        return hash_permuters_json[n_hashes]['a'], hash_permuters_json[n_hashes]['b']

    def _lists_to_vector(self, my_list, shape_into):
        if shape_into == "column":
            return np.asarray(my_list).reshape(-1, 1)
        elif shape_into == "row":
            return np.asarray(my_list).reshape(1, -1)
        raise ValueError(
            f" reshape_lists_into_vectors - Invalid value. Allowed values are: column (-1, 1) or rows (1, -1)")

    def _vectorize(self, texts):
        self.cv.fit(texts)
        vectors = self.cv.transform(texts).tolil().rows
        vectors_mask = (np.vectorize(len)(vectors) > 0)
        vectors = vectors[vectors_mask]
        texts = list(compress(texts, vectors_mask))
        indexed_texts = list(enumerate(texts))
        return indexed_texts, vectors

    def _create_minhash(self, row):
        A = self._lists_to_vector(self._a, 'column')
        X = self._lists_to_vector(row, 'row')
        B = self._lists_to_vector(self._b, 'column')
        hashes = np.matmul(A, X) + B
        hashes %= self.prime
        return hashes.min(axis=1)

    def fit_transform(self, texts:list[str]):
        """
        This function transforms a list of texts its signature of minhashes
        :return: a numpy matrix of shape (len(texts), n_hashes)
        """
        # 1) Fit Vectorize
        indexed_texts, vectors = self._vectorize(texts)
        # 2) Tranform to MinHash
        signatures = [self._create_minhash(row) for row in vectors]

        return np.array(signatures).T, indexed_texts

    def _find_similar_index_values(self, vector):
        vec = np.concatenate((vector.reshape(-1, 1), np.arange(vector.shape[0]).reshape(-1, 1)), axis=1)
        res = vec[vec[:, 0].argsort(kind='mergesort')]
        result = []
        current_val = res[0, 0]
        cache = []
        for i in range(1, res.shape[0]):
            if res[i, 0] == current_val:
                pass
            elif cache:
                cache_combined = np.array(list(itertools.product(cache, cache)))
                cache_combined = cache_combined[cache_combined[:, 0] < cache_combined[:, 1]]
                # cache_combined = np.hstack((cache_combined, np.ones((cache_combined.shape[0], 1))))
                cache_combined = cache_combined.astype(int)
                result.append(cache_combined)
                cache = []

            current_val = res[i, 0]
            cache.append(res[i, 1])

        result = np.concatenate(result)

        # return vals, res
        return result

    def _map_texts(self, counts, indexed_texts):
        df = pd.DataFrame(counts[:, :3], columns=['index_1', 'index_2', 'sim_hash'])
        df['text_1'] = df.index_1.map(dict(indexed_texts))
        df['text_2'] = df.index_2.map(dict(indexed_texts))
        return df

    def compare_hashes(self, signature_1, indexed_texts):
        """
        Provided 2 signatures, we want to compare which have similarity, and if yes, on how many hashes
        :return:
        """
        matching_combinations = []
        for i in range(self.n_hashes):
            X1 = signature_1[i]
            matching_combinations.extend(self._find_similar_index_values(X1))

        all_combinations = np.array(matching_combinations)
        unique, counts = np.unique(all_combinations, return_counts=True, axis=0)
        counts = counts.reshape(-1, 1)
        all_counts = np.concatenate((unique, counts/20), axis=1)
        all_counts = all_counts[np.argsort(all_counts[:, 2])[::-1]]

        df = self._map_texts(all_counts, indexed_texts)
        return df

if __name__ == "__main__":
    df = pd.DataFrame()
    model_nm = 'microsoft/deberta-v3-small'
    ds = Dataset.from_pandas(df)

    tokz = AutoTokenizer.from_pretrained(model_nm)

    print(tokz.tokenize("I am Samtosh, at your service"))

    def tok_func(x):
        return tokz(x["input"])

    tok_ds = ds.map(tok_func, batched=True)