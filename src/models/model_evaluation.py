import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_model(path, mode):
    model = pickle.load(open(path, mode))
    return model

def load_data(test_path):
    df = pd.read_csv(test_path)
    return df

def split_data(df):
    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values
    return X_test, y_test


def model_prediction(model, X_test,y_test):
    ypred = model.predict(X_test)
    accuracy = accuracy_score(y_test, ypred)
    precision = precision_score(y_test, ypred, average = 'weighted')
    recall = recall_score(y_test, ypred, average = 'weighted')
    metrics_dict = {
    'accuracy': round(accuracy*100,2),
    'precision':round(precision*100,2),
    'recall': round(recall*100,2)
    }
    return metrics_dict

def save_metrics(metrics_dict):

    with open ('./reports/metrics.json', 'w') as file:
        json.dump(metrics_dict, file, indent = 4)


def main():
    model =load_model('./models/model.pkl', 'rb')
    test_df = load_data('./data/processed/test_tfidf.csv')
    X_test, y_test = split_data(test_df)
    metrics_dict = model_prediction(model, X_test, y_test)
    save_metrics(metrics_dict)


if __name__ == '__main__':
    main()