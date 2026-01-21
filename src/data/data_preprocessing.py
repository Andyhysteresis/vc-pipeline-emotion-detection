import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
import os
import re

import logging

# configure the logger

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# create handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#connect the formatter to handler
console_handler.setFormatter(formatter)

#connect the handler to the logger
logger.addHandler(console_handler)


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')



lemy = WordNetLemmatizer()
stop = stopwords.words('english')
punc = string.punctuation

# Params file input
def load_param(params_path):
    pass

#load the data from folder

def load_data(data_path):
    try:
        df = pd.read_csv(data_path)    
        return df
    except FileNotFoundError:
        logger.error('File not found in the given url location')
        raise
    except pd.errors.ParserError:
        logger.error('There is some issue while parsing the file')
        raise


# transform the data
# Lemmatize
def lemmatization(text):
    text = text.split()
    text = [lemy.lemmatize(word) for word in text]
    return " ".join(text)

# Remove stopwords 
def remove_stop_words(text):
    text = [word for word in str(text).split() if word not in stop]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text
    
def lower_case(text):
    text = str(text)
    if text == 'nan' or text.strip() == '':
        return ''
    
    text = text.lower().strip()
    return text

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

    # Remove URLs
def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

# # Tokenize
# def tokenize_text(text):
#     text = nltk.word_tokenize(text)
#     return text

# # Keep only alphanumeric tokens
# def remove_special_characters(text):
#     text = [word for word in text if word.isalnum()]
#     return text



# transform the data

def transform_data(df):
    try:

        df['text'] = df['text'].apply(lower_case)
        logger.debug('converted to lower case')
        df['text'] = df['text'].apply(remove_stop_words)
        logger.debug('removed stop words')
        df['text'] = df['text'].apply(removing_numbers)
        logger.debug('removed numbers')
        df['text'] = df['text'].apply(removing_punctuations)
        logger.debug('removed punctuations')
        df['text'] = df['text'].apply(removing_urls)
        logger.debug('removed urls')
        df['text'] = df['text'].apply(lemmatization)
        logger.debug('performed lemmatization')
        return df
    except Exception as e:
        logger.error('Error during text transformation: %s', e)
        raise
 
def save_data(data_path, train_data, test_data):

    os.makedirs(data_path, exist_ok=True)

    train_data.to_csv(os.path.join(data_path, 'train_processed.csv'))
    logger.debug('train_data saved')
    test_data.to_csv(os.path.join(data_path, 'test_processed.csv'))
    logger.debug('test_data saved')

def main():
    try:

        train_data = load_data('./data/raw/train.csv')
        logger.debug('Train data loaded')
        test_data = load_data('./data/raw/test.csv')
        logger.debug('Test data loaded')

        train_processed_data = transform_data(train_data)
        logger.debug('Train data transformed')
        test_processed_data = transform_data(test_data)
        logger.debug('Test data transformed')

        data_path = os.path.join('data', 'interim')

        save_data(data_path, train_processed_data, test_processed_data)
    
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        raise

if __name__ == '__main__':
    main()