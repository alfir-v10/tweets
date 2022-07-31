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
        # prep = Preprocessing(path_data=path_data)
        # prep.run_preprocessing()
        # sentences = prep.df.iloc[:, 0].to_numpy()
        for file in listdir:
            start_time = time.time()
            vector = pd.read_csv(os.path.join(path_datasets, file), index_col=0)
            vector.drop_duplicates(inplace=True)
            for model in models:
                start_time_model = time.time()
                print(f"Running {model.__name__} on dataset by {file}")
                x_data = vector.iloc[:, 1:].values
                y_data = vector.iloc[:, 0].values

                x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
                clf = model()
                cv = GridSearchCV(clf, param_grid=models[model]['param_grid'],
                                  verbose=1, n_jobs=4, scoring='roc_auc_ovr',
                                  refit='roc_auc_ovr')
                cv.fit(x_train, y_train)
                print(f"Best params_: {cv.best_params_}\n"
                      f"Best roc_auc_score: {round(cv.best_score_, 3)}")
                y_pred = cv.predict(x_test)
                onehot_encoder = OneHotEncoder(sparse=False)
                y_test = onehot_encoder.fit_transform(y_test.reshape(len(y_test), 1))
                y_pred = onehot_encoder.fit_transform(y_pred.reshape(len(y_pred), 1))
                score = roc_auc_score(y_test, y_pred, multi_class='ovr')
                print('Test roc_auc_score: ', round(score, 3))
                diff_time = time.time() - start_time_model

                path_res = "fit_results.csv"
                conf_name = file[:-4]
                if not os.path.exists(path_res):
                    with open(path_res, 'w') as f:
                        f.write("dataset, model, score, time execution\n")
                with open(path_res, "a") as f:
                    f.write(f"{conf_name}, {model.__name__}, {round(score, 3)}, {round(diff_time, 2)}\n")

                if not os.path.exists("models/"):
                    os.mkdir("models/")
                dump(cv.best_estimator_, f"models/{conf_name} - {model.__name__}.joblib")
