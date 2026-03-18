import pandas as pd
import re
import logging


def clean_text(text):
    # Normaliser
    
    text = text.str.lower()
    # Filtrage de la ponctuation (. , .., ..., )
    text = re.sub(r"\.+", '', text)

    text = re.sub(r"/", ' ', text)


    # Filtrage des chiffres
    text= re.sub(r"[0-9]+", '', text)

    return text


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info('cleaning data set')

    X_train_path = "data/processed_data/X_train.csv"
    X_test_path = "data/processed_data/X_test.csv"
    X_train = pd.read_csv(X_train_path, sep=",")
    X_test = pd.read_csv(X_test_path, sep=",")
    X_train['text'] = X_train['summary'] + " " + X_train['reviewText']
    X_test['text'] = X_test['summary'] + " " + X_test['reviewText']
    X_train['text'] = X_train['text'].apply(lambda x: clean_text(x), axis=1)
    X_test['text'] = X_test['text'].apply(lambda x: clean_text(x), axis=1)
    X_train.to_pickle("data/processed_data/X_train.pickle")
    X_test.to_pickle("data/processed_data/X_test.pickle")