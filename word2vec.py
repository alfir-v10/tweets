import pandas as pd
from gensim.models import word2vec
import numpy as np
from pprint import pprint
import gensim
import wget
import gzip
import shutil
import gensim.downloader
from preprocessing import Preprocessing

if __name__ == "__main__":
    w2v = word2vec.Word2Vec()
    # download bin.gz from: https://code.google.com/archive/p/word2vec/
    w2v = gensim.downloader.load('word2vec-google-news-300')

    # w2v_vocab = set(w2v)

    # print("Loaded {} words in vocabulary".format()

    pr = Preprocessing()
    pr.run_preprocessing()
    sentences = pr.df.iloc[:, 0].values
    # sentences =
    # sentences = ["I'm happy to shop in Walmart and buy a Google phone",
    #              "In today's demo we'll look at Office and Word from microsoft",
    #              "Tech companies like Apple with their iPhone are the new cool",
    #              "Yesterday I went swimming",
    #              "Pepsi is drunk by a New Generation",
    #              "Bob has an Android Nexus 5 for his telephone",
    #              "Alice drinks coffee every morning",
    #              "I want to drink a coke and eat something",
    #              "You'll be happier if you take a swim",
    #              "This is a really long sentence that hopefully doesn't get a very high score just because it has lots of words in it!"]

    target_sentence = "Microsoft smartphones are the latest buzz"

    most_common = []
    for i, target_sentence in enumerate(sentences):
        if not len(sentences) - 1:
            break
        sentences_similarity = np.zeros(len(sentences) - 1)
        target_sentence_words = [w for w in target_sentence.split() if w in w2v]

        for idx, sentence in enumerate(sentences):
            if target_sentence != sentence:
                sentence_words = [w for w in sentence.split() if w in w2v]
                sim = 0
                if sentence_words and target_sentence_words:
                    sim = w2v.n_similarity(target_sentence_words, sentence_words)
                sentences_similarity[idx - 1] = sim
        sentences = sentences[1:]
        result = list(zip(sentences_similarity, sentences))
        result.sort(key=lambda item: item[0], reverse=True)
        most_common.append((target_sentence, result[0][1], result[0][0]))
        print("Target:", target_sentence)
        # pprint(result)
    most_common.sort(key=lambda item: item[2], reverse=True)
    with open("most_common_w2v.txt", 'w') as f:
        for most in most_common:
            f.write(f"{most[0], most[1], most[2]}\n")

    # print(w2v.wmdistance(target_sentence, sentences[0]))