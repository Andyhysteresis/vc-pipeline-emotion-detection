import numpy as np
import pandas as pd
import os

import yaml

import warnings as w
w.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

import logging

# configure logging

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# creating handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#creating formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#connecting formatter with handler

console_handler.setFormatter(formatter)

# coonecting the handler to the logger
logger.addHandler(console_handler)

def load_params(params_path: str) -> float:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        logger.debug('test size retrieved')
        return test_size
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except yaml.YAMLError as e:
        logger.error('yaml error')
        raise
    except Exception as e:
        logger.error('some error occurred')
        raise


def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, encoding ='latin1')
        return df

    except FileNotFoundError:
        logger.error('File not found in this location')
        raise

    except pd.errors.ParserError:
        logger.error('File has some parsing issues. Pls check carefully')
        raise

    except Exception as e:
        logger.error('Some other error occurred')
        raise


def process_data(df: pd.DataFrame) -> pd.DataFrame:

    try:
        cols =['textID', 'selected_text', 'Time of Tweet',
       'Age of User', 'Country', 'Population -2020', 'Land Area (Km²)',
       'Density (P/Km²)']
        df.drop(columns=cols, inplace=True)
        logger.debug('Droppeed the TWEET_ID columns for the analysis')
        # final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        # final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        df['sentiment'] = df['sentiment'].replace({'positive': 1, 'neutral': 0, 'negative': -1})
        logger.debug('DataFrame is made and the sentiment values are replaced with numbers')
        
        return df

    except KeyError as e:
        logger.error('The key value of the column name is missing/wrong')
        raise

    except Exception as e:
        logger.error('some other error occurred')
        raise

def save_data(data_path, train_data, test_data):

    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'))
        logger.debug('Train data saved')
        test_data.to_csv(os.path.join(data_path, 'test.csv'))
        logger.debug('Test data saved')
    
    except Exception as e:
        logger.error('some unexpected error occurred during saving of data/file')
        raise


def main():
    test_size = load_params('params.yaml')
    
    df = read_data(r'C:/MLOps/version_control_sentiment_analysis/sentiment_analysis_dataset.csv')
    final_df = process_data(df)

    train_data, test_data = train_test_split(final_df, test_size = test_size, random_state = 42)
    data_path = os.path.join('data', 'raw')
    save_data(data_path, train_data, test_data)


if __name__ == '__main__':
    main()



