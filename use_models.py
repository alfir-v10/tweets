import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    path_models = './models'
    path_datasets = "./datasets"
    listdir_datasets = os.listdir(path_datasets)
    listdir_models = os.listdir(path_models)
    model = load(os.path.join(path_models, listdir_models[0]))
    vector = pd.read_csv(os.path.join(path_datasets, listdir_datasets[0]), index_col=0)
    for i in range(10):
        x_test = vector.iloc[i, 1:][vector.iloc[i, 1:] > 0].index
        y_test = vector.iloc[i, 1]
        print(x_test)
        x_test = vector.iloc[i, 1:].values
        y_pred = model.predict(x_test.reshape(1, -1))
        print(y_pred, y_test)
        print("\n")
    # onehot_encoder = OneHotEncoder(sparse=False)
    # print(y_test[0], y_pred[0], x_test[0])
    # y_test = onehot_encoder.fit_transform(y_test.reshape(len(y_test), 1))
    # y_pred = onehot_encoder.fit_transform(y_pred.reshape(len(y_pred), 1))
    # score = roc_auc_score(y_test, y_pred, multi_class='ovr')
    # print('Test roc_auc_score: ', round(score, 3))
    #
    pass