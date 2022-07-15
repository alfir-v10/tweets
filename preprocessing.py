import pandas as pd
import os
from config import params
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from textblob import TextBlob
from spellchecker import SpellChecker
import re
import time
import numpy as np
from collections import Counter, defaultdict


C_GREEN = "\033[32m "
C_BLUE = "\033[34m "
C_RED = "\033[31m "
C_WHITE = "\033[37m "
C_YELLOW = "\033[33m "
C_END = " \033[0m"


class Preprocessing:
    def __init__(self, path_data: str = './data/', params: dict = None):
        self.docs_list = None
        self.vector_columns = None
        self.path_data = path_data
        self.encoding = None
        self.stemm = None
        self.lemm = None
        self.miss = None
        self.punct = None
        self.stop_words = None
        self.descr = None
        self.lower = None
        self.nbr_ex = None
        if params:
            self.params = params
            if params['coder']:
                self.encoding = params['coder']  # one_hot or bag_of_words or tfidf
            if params['stemming']:
                self.stemm = params['stemming']
            if params['lemmatization']:
                self.lemm = params['lemmatization']
            if params['misspellings']:
                self.miss = params['misspellings']
            if params['punctuation']:
                self.punct = params['punctuation']
            if params['stop_words']:
                self.stop_words = params['stop_words']
            if params['description']:
                self.descr = params['description']
            if params['lower_case']:
                self.lower = params['lower_case']
            if params['chose_example']:
                self.nbr_ex = params['chose_example']
        self.df = None
        self.vector = None
        self.corpus = None
        self.corpus_counter = None

    def _time_execution(func_name):
        def print_time(self, *args):
            start_time = time.process_time()
            func_name(self, *args)
            end_time = time.process_time()
            print(f"{C_RED}Elapsed time during the whole {func_name.__name__} in seconds: {C_END}", end_time-start_time)
        return print_time

    def _print_decorator(func_name):
        def print_example(self, *args):
            if isinstance(self.nbr_ex, int) and 0 <= self.nbr_ex < self.df.shape[0]:
                print(f"{C_BLUE}Before {func_name.__name__}: {C_END}", self.df.iloc[self.nbr_ex, 2])
            # print(f"{when} {name}: ", )
            start_time = time.process_time()
            func_name(self, *args)
            if isinstance(self.nbr_ex, int) and 0 <= self.nbr_ex < self.df.shape[0]:
                print(f"{C_GREEN}After {func_name.__name__}: {C_END}", self.df.iloc[self.nbr_ex, 2])
            end_time = time.process_time()
            print(f"{C_RED}Elapsed time during the whole {func_name.__name__} in seconds: {C_END}", end_time-start_time)
        return print_example

    @_print_decorator
    def run_tokenazation(self):
        self.df.iloc[:, 2] = self.df.iloc[:, 2].agg(word_tokenize)

    @_print_decorator
    def run_stemming(self, stemmer=LancasterStemmer):
        self.df.iloc[:, 2] = self.df.iloc[:, 2].agg(lambda text: [stemmer().stem(word) for word in text])

    @_print_decorator
    def run_lemmatization(self, lemmatize=WordNetLemmatizer):
        self.df.iloc[:, 2] = self.df.iloc[:, 2].agg(lambda text: [lemmatize().lemmatize(word) for word in text])

    @_print_decorator
    def run_punctuation(self):
        self.df.iloc[:, 2] = self.df.iloc[:, 2].agg(lambda text: ["".join([c for c in word
                                                                           if (c in string.ascii_letters)
                                                                           or (c in string.digits)
                                                                           or (c in string.whitespace)])
                                                                  for word in text])
        self.df.iloc[:, 2] = self.df.iloc[:, 2].agg(lambda text: [word for word in text
                                                                  if word.isalpha()])

    @_print_decorator
    def run_lower_case(self):
        self.df.iloc[:, 2] = self.df.iloc[:, 2].agg(lambda text: [word.lower() for word in text])

    @_print_decorator
    def run_misspellings(self):
        spell = SpellChecker()
        self.df.iloc[:, 2] = self.df.iloc[:, 2].agg(lambda text: [TextBlob(word).correct().string
                                                                  if word in spell.unknown(text) else word
                                                                  for word in text])

    @_print_decorator
    def run_stop_words(self):
        self.df.iloc[:, 2] = self.df.iloc[:, 2].agg(lambda text: [word for word in text if word not in self.stop_words])

    def read_file(self):
        df_negative = pd.read_csv(os.path.join(self.path_data, 'processedNegative.csv')).T
        df_neutral = pd.read_csv(os.path.join(self.path_data, 'processedNeutral.csv')).T
        df_positive = pd.read_csv(os.path.join(self.path_data, 'processedPositive.csv')).T
        df_positive['label'] = 0
        df_negative['label'] = 1
        df_neutral['label'] = 2
        # print(df_positive.shape, df_negative.shape, df_neutral.shape)
        self.df = pd.concat([df_positive, df_neutral, df_negative])
        self.df.reset_index(inplace=True)
        self.df['tweets'] = self.df.iloc[:, 0]

    def create_vector(self):
        self.df['copy'] = self.df.iloc[:, 2].agg(lambda text: " ".join(text))
        documents = " ".join(self.df['copy'].values).split(" ")
        self.corpus = sorted(list(set(documents)))
        self.corpus_counter = Counter(documents)
        len_corpus = len(self.corpus)
        print(f"Corpus volume: {len_corpus}")
        self.df.drop(['copy'], axis=1, inplace=True)

        df_zeros = pd.DataFrame(data=np.zeros((self.df.shape[0], len_corpus)),
                                columns=self.corpus, dtype=int)
        self.vector = pd.concat([self.df.iloc[:, 1], df_zeros], axis=1)
        self.vector_columns = self.vector.columns[1:]
        self.docs_list = self.df.iloc[:, 2].values.tolist()

    @_time_execution
    def one_hot_encoding(self):
        print(f"{C_YELLOW}Start one-hot encoding {C_END}")
        self.create_vector()
        for i, tweet in enumerate(self.docs_list):
            for word in tweet:
                if word in self.vector_columns:
                    if not self.vector.loc[i, word]:
                        self.vector.loc[i, word] += 1

    @_time_execution
    def bag_of_words_encoding(self):
        print(f"{C_YELLOW}Start bag-of-words encoding {C_END}")
        self.create_vector()
        for i, tweet in enumerate(self.docs_list):
            for word in tweet:
                self.vector.loc[i, word] += 1

    @_time_execution
    def tfidf_encoding(self):
        print(f"{C_YELLOW}Start tf-idf encoding {C_END}")
        self.create_vector()
        for i, tweet in enumerate(self.docs_list):
            word_freq = Counter(tweet)
            for word in word_freq:
                self.vector.loc[i, word] = 1 + np.log(word_freq[word])

        for i, tweet in enumerate(self.docs_list):
            for word in tweet:
                self.vector.loc[i, word] = self.vector.loc[i, word] * np.log(
                    1 + self.vector.shape[0] / np.sum([1 for k in self.df.iloc[:, 2].values.tolist() if word in k]))

    def run_encoding(self):
        if self.encoding == 'one_hot':
            self.one_hot_encoding()
        if self.encoding == 'bag_of_words':
            self.bag_of_words_encoding()
        if self.encoding == 'tfidf':
            self.tfidf_encoding()

    @_time_execution
    def run_preprocessing(self):
        self.read_file()
        self.run_tokenazation()
        if self.descr:
            print("Approach: ", self.descr)

        if not self.stemm and self.lower:
            self.run_lower_case()

        if self.punct:
            self.run_punctuation()

        if self.stop_words:
            if isinstance(self.stop_words, list):
                print(f"{C_YELLOW}Use your stopwords list and stopwords from nltk{C_END}")
                self.stop_words = self.stop_words + stopwords.words('english')
                self.run_stop_words()
            if isinstance(self.stop_words, bool):
                print(f"{C_YELLOW}Use stopwords from nltk{C_END}")
                self.stop_words = stopwords.words('english')
                self.run_stop_words()

        if self.miss:
            self.run_misspellings()
        if self.stemm:
            self.run_stemming()
        if self.lemm:
            self.run_lemmatization()
        if self.encoding:
            self.run_encoding()


if __name__ == "__main__":
    path = './data/'
    if not os.path.exists("datasets"):
        os.mkdir("datasets")
    for config in params:
        prep = Preprocessing(path_data=path,
                             params=params[config])
        prep.run_preprocessing()
        vector = prep.vector
        vector.to_csv(f"datasets/{config}.csv")
        print(f"DataFrame {config} is Done!")