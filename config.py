from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import numpy as np

params = {
    #for test
    # 'test': {
    #         'coder': 'one_hot',
    #         'stemming': True,
    #         'lemmatization': False,
    #         'misspellings': False,
    #         'stop_words': True,
    #         'punctuation': True,
    #         'lower_case': True,
    #         'description': 'testing',
    #         'chose_example': None
    #     },
    # one_hot
    '00': {
        'coder': 'one_hot',
        'stemming': False,
        'lemmatization': False,
        'misspellings': False,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'chose_example': None,
        'description': '0 or 1, if the word exist + just tokenization',
    },
    '01': {
        'coder': 'one_hot',
        'stemming': True,
        'lemmatization': False,
        'misspellings': False,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'chose_example': None,
        'description': '0 or 1, if the word exist + stemming',
    },
    '02': {
        'coder': 'one_hot',
        'stemming': False,
        'lemmatization': True,
        'misspellings': False,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'chose_example': None,
        'description': '0 or 1, if the word exist + lemmatization'
    },
    '03': {
        'coder': 'one_hot',
        'stemming': True,
        'lemmatization': False,
        'misspellings': True,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'chose_example': None,
        'description': '0 or 1, if the word exist + stemming + misspellings'
    },
    '04': {
        'coder': 'one_hot',
        'stemming': False,
        'lemmatization': True,
        'misspellings': True,
        'chose_example': None,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'description': '0 or 1, if the word exist + lemmatization + misspellings'
    },
    # '05': {
    #     'coder': 'one_hot',
    #     'stemming': True,
    #     'lemmatization': True,
    #     'misspellings': True,
    #     'chose_example': None,
    #     'stop_words': True,
    #     'punctuation': True,
    #     'lower_case': True,
    #     'description': '0 or 1, if the word exist + any other ideas of preprocessing'
    # },

    # bag_of_words
    '10': {
        'coder': 'bag_of_words',
        'stemming': False,
        'lemmatization': False,
        'misspellings': False,
        'chose_example': None,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'description': 'bag_of_words + just tokenization',
    },
    '11': {
        'coder': 'bag_of_words',
        'stemming': True,
        'lemmatization': False,
        'chose_example': None,
        'misspellings': False,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'description': 'bag_of_words + stemming',
    },
    '12': {
        'coder': 'bag_of_words',
        'stemming': False,
        'lemmatization': True,
        'chose_example': None,
        'misspellings': False,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'description': 'bag_of_words + lemmatization'
    },
    '13': {
        'coder': 'bag_of_words',
        'stemming': True,
        'lemmatization': False,
        'misspellings': True,
        'chose_example': None,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'description': 'bag_of_words + stemming + misspellings'
    },
    '14': {
        'coder': 'bag_of_words',
        'stemming': False,
        'lemmatization': True,
        'misspellings': True,
        'chose_example': None,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'description': 'bag_of_words + lemmatization + misspellings'
    },
    # '15': {
    #     'coder': 'bag_of_words',
    #     'stemming': True,
    #     'chose_example': None,
    #     'lemmatization': True,
    #     'misspellings': True,
    #     'stop_words': True,
    #     'punctuation': True,
    #     'lower_case': True,
    #     'description': 'bag_of_words + any other ideas of preprocessing'
    # },

    # tfidf
    '20': {
        'coder': 'tfidf',
        'stemming': False,
        'lemmatization': False,
        'misspellings': False,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'chose_example': None,
        'description': 'TFIDF + just tokenization',
    },
    '21': {
        'coder': 'tfidf',
        'stemming': True,
        'lemmatization': False,
        'misspellings': False,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'chose_example': None,
        'description': 'TFIDF + stemming',
    },
    '22': {
        'coder': 'tfidf',
        'stemming': False,
        'lemmatization': True,
        'misspellings': False,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'chose_example': None,
        'description': 'TFIDF + lemmatization'
    },
    '23': {
        'coder': 'tfidf',
        'stemming': True,
        'lemmatization': False,
        'misspellings': True,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'chose_example': None,
        'description': 'TFIDF + stemming + misspellings'
    },
    '24': {
        'coder': 'tfidf',
        'stemming': False,
        'lemmatization': True,
        'chose_example': None,
        'misspellings': True,
        'stop_words': True,
        'punctuation': True,
        'lower_case': True,
        'description': 'TFIDF + lemmatization + misspellings'
    },
    # '25': {
    #     'coder': 'tfidf',
    #     'stemming': False,
    #     'lemmatization': True,
    #     'misspellings': True,
    #     'stop_words': True,
    #     'punctuation': True,
    #     'lower_case': True,
    #     'chose_example': None,
    #     'description': 'TFIDF + any other ideas of preprocessing'
    # },
}

models = {
        LogisticRegression:
            {
                'param_grid':
                    {
                        'penalty': ['l2'],
                        'C': [0.01, 0.1, 1,], # 0.5, 1, 2
                        'max_iter': [25, 50, 100], # 100, 200
                        'multi_class': ['ovr', ],
                    },
            },
        DecisionTreeClassifier:
            {
                'param_grid': {
                    'criterion': ['gini', 'entropy', 'log_loss'], #'entropy', 'log_loss'
                    'max_depth': [3, 5, 10], #20, 30
                    'min_samples_split': [2, 4, 6], #4, 6
                    'min_samples_leaf': [1, 3, 5] #3, 5
                }
            },
        RandomForestClassifier:
            {
                'param_grid': {
                    "n_estimators": [50, 100, 200, 300], #
                    "criterion": ['gini'], #'entropy', 'log_loss'
                    "max_depth": [3, 4, 6], #20, 30
                    'min_samples_split': [2, 4, 6], #4, 6
                    'min_samples_leaf': [1,3, 5], #3, 5
                }
            },
        GradientBoostingClassifier:
            {
                'param_grid':
                    {
                        "loss": ["log_loss", 'deviance', 'exponential'], #'deviance', 'exponential'
                        "learning_rate" : [0.1, 0.05, 0.01],
                        "n_estimators" : [100, 200, 300],
                        "criterion": ['friedman_mse', "squared_error", "mse"], # squared_error, mse
                        "min_samples_split": [2, 4, 6],#
                        "min_samples_leaf": [1, 3, 5],  #
                        "max_depth": [3, 6, 9],  #
                        "n_iter_no_change": [3,],  #
                    }
            },
        XGBClassifier:
            {
                'param_grid': {
                    'max_depth': [2, ],
                    'n_estimators': [60, ],
                    'learning_rate': [0.1, ],
                }
            },
}
