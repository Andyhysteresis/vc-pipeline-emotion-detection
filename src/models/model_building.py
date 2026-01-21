import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier

def load_params(params_path):
    params = yaml.safe_load(open('params.yaml','r'))['model_building']
    n_estimators = params['n_estimators']
    learning_rate = params['learning_rate']
    return n_estimators, learning_rate

# fetch the data from data/processed
def load_data(path):
    df = pd.read_csv(path)
    return df

def split_data(df):
    X_train = df.iloc[:, :-1].values
    y_train = df.iloc[:, -1].values
    return X_train, y_train

# Define and train the model
def model_define(X_train, y_train, n_estimators, learning_rate):
    model = GradientBoostingClassifier(n_estimators = n_estimators, learning_rate=learning_rate)
    model.fit(X_train,y_train)
    return model


# Save the model
def model_save(model):
    pickle.dump(model, open('./models/model.pkl','wb'))

def main():
    n_estimators, learning_rate = load_params('params.yaml')
    train_data = load_data('./data/processed/train_tfidf.csv')
    X_train, y_train = split_data(train_data)
    model = model_define(X_train, y_train, n_estimators, learning_rate)
    model_save(model)

if __name__ == '__main__':
    main()



