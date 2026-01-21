import numpy as np
import pandas as pd
import os
import yaml
from sklearn.feature_extraction.text import CountVectorizer
import logging

# configure logging
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

#create handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# connect the handler with the formatter
console_handler.setFormatter(formatter)

# connect the handler with logger
logger.addHandler(console_handler)

def load_params(params_path):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('The file is not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('There is some issue with yaml file: %s', e)
        raise
    except Exception as e:
        logger.error('Some other error occured during loading the param file: %s', e)
        raise
        
# fetch the data from data/processed
def load_data(path):
    try:
        df = pd.read_csv(path)
        df.fillna('', inplace = True)
        logger.debug('Data loaded and NaN values are filled from %s', path)
        return df

    except FileNotFoundError:
        logger.error('The file is not found: %s', path)
        raise
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading the data: %s', e)
        raise


def Vectorization(train_data: pd.DataFrame, test_data:pd.DataFrame, max_features: int)-> tuple:
    
    try:
        # Apply TFIDF vectorizer
        vectorizer = CountVectorizer(max_features = max_features)
        logger.debug('created Vectorizer object')

        X_train = train_data['text'].values
        logger.debug('Converted train_data in X_train numpy array values')
        y_train = train_data['sentiment'].values
        logger.debug('Converted train_data in y_train numpy array values')

        X_test = test_data['text'].values
        logger.debug('Converted test_data in X_test numpy array values')
        y_test = test_data['sentiment'].values
        logger.debug('Converted test_data in y_test numpy array values')


        #Fit the vectorizer onto the training data
        X_train_bow = vectorizer.fit_transform(X_train)
        logger.debug('fitted and transformed the Vectorizer object on X_train')

        #Fit the vectorizer onto the test  data
        X_test_bow = vectorizer.transform(X_test)
        logger.debug('transformed the Vectorizer object on X_test')

        # Making the train_df
        train_df = pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out())
        train_df['label'] = y_train
        logger.debug('Applied Vectorizer and transformed the training data')

        # Making the test_df
        test_df = pd.DataFrame(X_test_bow.toarray(), columns=vectorizer.get_feature_names_out())
        test_df['label'] = y_test
        logger.debug('Applied Vectorizer and transformed the test data')

        return train_df, test_df
    
    except Exception as e:
        logger.error('Unexpected error during Vector transformation: %s', e)
        raise

# store the data inside data/features

def save_data(data_path, train_df, test_df):
    try:
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, 'train_bow.csv'))
        test_df.to_csv(os.path.join(data_path, 'test_bow.csv'))
    except Exception as e:
        logger.error('Some unexpected error occurred during saving the file: %s', e)
        raise


def main():
    params = load_params('params.yaml')
    max_features = params['feature_engineering']['max_features']
    logger.debug('max features is set')

    train_data = load_data('./data/interim/train_processed.csv')
    logger.debug('train data is loaded')

    test_data = load_data('./data/interim/test_processed.csv') 
    logger.debug('test data is loaded')     

    train_bow, test_bow = Vectorization(train_data, test_data, max_features)

    data_path = os.path.join('data', 'processed')
    save_data(data_path, train_bow,test_bow)
    logger.debug('Train and test data is saved')


if __name__ == '__main__':
    main()
