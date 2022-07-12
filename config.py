from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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
    #         'chose_example': 45
    #     },
    # one_hot
    '00': {
        'coder': 'one_hot',
        'stemming': False,
        'lemmatization': False,
        'misspellings': False,
        'stop_words': False,
        'punctuation': False,
        'lower_case': False,
        'chose_example': None,
        'description': '0 or 1, if the word exist + just tokenization',
    },
    '01': {
        'coder': 'one_hot',
        'stemming': True,
        'lemmatization': False,
        'misspellings': False,
        'stop_words': False,
        'punctuation': False,
        'lower_case': False,
        'chose_example': None,
        'description': '0 or 1, if the word exist + stemming',
    },
    '02': {
        'coder': 'one_hot',
        'stemming': False,
        'lemmatization': True,
        'misspellings': False,
        'stop_words': False,
        'punctuation': False,
        'lower_case': False,
        'chose_example': None,
        'description': '0 or 1, if the word exist + lemmatization'
    },
    '03': {
        'coder': 'one_hot',
        'stemming': True,
        'lemmatization': False,
        'misspellings': True,
        'stop_words': False,
        'punctuation': False,
        'lower_case': False,
        'chose_example': None,
        'description': '0 or 1, if the word exist + stemming + misspellings'
    },
    '04': {
        'coder': 'one_hot',
        'stemming': False,
        'lemmatization': True,
        'misspellings': True,
        'chose_example': None,
        'stop_words': False,
        'punctuation': False,
        'lower_case': False,
        'description': '0 or 1, if the word exist + lemmatization + misspellings'
    },
    '05': {
        'coder': 'one_hot',
        'stemming': True,
        'lemmatization': True,
        'misspellings': True,
        'stop_words': True,
        'chose_example': None,
        'punctuation': True,
        'lower_case': True,
        'description': '0 or 1, if the word exist + any other ideas of preprocessing'
    },

    # bag_of_words
    '10': {
        'coder': 'bag_of_words',
        'stemming': False,
        'lemmatization': False,
        'misspellings': False,
        'chose_example': None,
        'stop_words': False,
        'punctuation': False,
        'lower_case': False,
        'description': 'bag_of_words + just tokenization',
    },
    '11': {
        'coder': 'bag_of_words',
        'stemming': True,
        'lemmatization': False,
        'chose_example': None,
        'misspellings': False,
        'stop_words': False,
        'punctuation': False,
        'lower_case': False,
        'description': 'bag_of_words + stemming',
    },
    '12': {
        'coder': 'bag_of_words',
        'stemming': False,
        'lemmatization': True,
        'chose_example': None,
        'misspellings': False,
        'stop_words': False,
        'punctuation': False,
        'lower_case': False,
        'description': 'bag_of_words + lemmatization'
    },
    '13': {
        'coder': 'bag_of_words',
        'stemming': True,
        'lemmatization': False,
        'misspellings': True,
        'stop_words': False,
        'chose_example': None,
        'punctuation': False,
        'lower_case': False,
        'description': 'bag_of_words + stemming + misspellings'
    },
    '14': {
        'coder': 'bag_of_words',
        'stemming': False,
        'lemmatization': True,
        'misspellings': True,
        'stop_words': False,
        'chose_example': None,
        'punctuation': False,
        'lower_case': False,
        'description': 'bag_of_words + lemmatization + misspellings'
    },
    '15': {
        'coder': 'bag_of_words',
        'stemming': True,
        'chose_example': None,
        'lemmatization': True,
        'misspellings': True,
        'stop_words': True,
        'punctuation': True,
        'lower_case': False,
        'description': 'bag_of_words + any other ideas of preprocessing'
    },

    # tfidf
    '20': {
        'coder': 'tfidf',
        'stemming': False,
        'lemmatization': False,
        'misspellings': False,
        'stop_words': False,
        'punctuation': False,
        'lower_case': False,
        'chose_example': None,
        'description': 'TFIDF + just tokenization',
    },
    '21': {
        'coder': 'tfidf',
        'stemming': True,
        'lemmatization': False,
        'misspellings': False,
        'stop_words': False,
        'lower_case': False,
        'punctuation': False,
        'chose_example': None,
        'description': 'TFIDF + stemming',
    },
    '22': {
        'coder': 'tfidf',
        'stemming': False,
        'lemmatization': True,
        'misspellings': False,
        'stop_words': False,
        'lower_case': False,
        'punctuation': False,
        'chose_example': None,
        'description': 'TFIDF + lemmatization'
    },
    '23': {
        'coder': 'tfidf',
        'stemming': True,
        'lemmatization': False,
        'misspellings': True,
        'stop_words': False,
        'punctuation': False,
        'lower_case': False,
        'chose_example': None,
        'description': 'TFIDF + stemming + misspellings'
    },
    '24': {
        'coder': 'tfidf',
        'stemming': False,
        'lemmatization': True,
        'chose_example': None,
        'misspellings': True,
        'stop_words': False,
        'punctuation': False,
        'lower_case': False,
        'description': 'TFIDF + lemmatization + misspellings'
    },
    '25': {
        'coder': 'tfidf',
        'stemming': False,
        'lemmatization': True,
        'misspellings': True,
        'stop_words': True,
        'punctuation': True,
        'chose_example': None,
        'lower_case': True,
        'description': 'TFIDF + any other ideas of preprocessing'
    },
}

models = [
        LogisticRegression,
        DecisionTreeClassifier,
        RandomForestClassifier,
        # GradientBoostingClassifier,
        # AdaBoostClassifier,
        # CatBoostClassifier,
        XGBClassifier,
        ]
