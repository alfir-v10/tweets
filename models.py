import os
from os import path
import time
import string
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from collections import Counter, defaultdict
import logging
logger = logging.getLogger("tweets_log.log")
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from joblib import dump, load

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')



from preprocessing import Preprocessing
from config import params, models
import datetime

def save_result(results, path=None):
    df_res = pd.DataFrame(results)
    if path:
        df_res.to_csv(path)
    else:
        df_res.to_csv(f"{datetime.datetime.now().date()} "
                      f"{datetime.datetime.now().hour}-{datetime.datetime.now().minute}.csv")


if __name__ == "__main__":
    path_data = './data/'
    path_datasets = "./datasets"
    listdir = os.listdir(path_datasets)
    if listdir:
        prep = Preprocessing(path_data=path_data)
        prep.run_preprocessing()
        sentences = prep.df.iloc[:, 0].to_numpy()
        for file in listdir:
            start_time = time.time()
            vector = pd.read_csv(os.path.join(path_datasets, file), index_col=0)
            vector.drop_duplicates(inplace=True)

            # corpus_len = len(vector.shape[1])
