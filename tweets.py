import os
from os import path
import time
import string
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image



import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')



from preprocessing import Preprocessing
from config import params, models
import datetime

def save_result(results, path='results.csv'):
    df_res = pd.DataFrame(results)
    df_res.to_csv(str(datetime.datetime.now().date()) + path)

if __name__ == "__main__":
    path = './data/'
    results = {}
    try:
        for config in params:
            prep = Preprocessing(path_data=path,
                                 params=params[config])
            prep.run_preprocessing()
            vector = prep.vector
            result = {}
            for model in models:
                print(f"Running {model.__name__} on dataset by {prep.descr}")
                x_data = vector.iloc[:, 1:].values
                y_data = vector.iloc[:, 0].values
                x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
                clf = model()
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                score = accuracy_score(y_test, y_pred)
                print('Test Accuracy: ', round(score, 3))
                result.update({model.__name__: score})
            results.update({config: result})
            save_result(results)
    except Exception as e:
        print(e)
        save_result(results)
