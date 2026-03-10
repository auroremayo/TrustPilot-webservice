import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import tqdm
from tqdm import tqdm
import logging
from nltk.stem import WordNetLemmatizer


def stop_words_filtering(mots) :
    try:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        tokens = []
        for mot in mots:
            if mot not in stop_words:
                tokens.append(mot)
        return tokens
    except Exception as e:
        logger.error(e)
        raise


def tokenize_text(X_train_path, X_test_path):
    try:
        X_train = pd.read_pickle(X_train_path)
        X_test = pd.read_pickle(X_test_path)

        nltk.download('punkt_tab')
        X_train['text_tokenized'] = X_train['text'].apply(lambda x: word_tokenize(x, language='english'))
        X_test['text_tokenized'] = X_test['text'].apply(lambda x: word_tokenize(x, language='english'))

        X_train['text_tokenized'] = X_train['text_tokenized'].apply(stop_words_filtering)
        X_test['text_tokenized'] = X_test['text_tokenized'].apply(stop_words_filtering)
        return X_train, X_test
    
    except Exception as e:
        logger.error(e)
        raise
    

def lemmatisation(mots) :
    try:
        nltk.download('wordnet')
        wordnet_lemmatizer = WordNetLemmatizer()
        sortie = []
        for string in mots :
            radical = wordnet_lemmatizer.lemmatize(string)
            if (radical not in sortie) : sortie.append(radical)
        return sortie
    except Exception as e :
        logger.error(e)
        raise

if __name__ == '__main__':
    try:
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)

        logger = logging.getLogger(__name__)
        logger.info('cleaning data set')

        X_train_path = "X_train.csv"
        X_test_path = "X_test.csv"
        X_train, X_test = tokenize_text(X_train_path, X_test_path)
        
        logger.info('lemmatizing data set')
        X_train['lemmes'] = X_train['text_tokenized'].apply(lemmatisation)
        X_test['lemmes'] = X_test['text_tokenized'].apply(lemmatisation)

        for i in X_train.index :
            X_train.loc[i, 'lemmes'] = ' '.join(X_train.loc[i, 'lemmes'])

        for i in X_test.index :
            X_test.loc[i, 'lemmes'] = ' '.join(X_test.loc[i, 'lemmes'])

        logger.info('saving lemmatized data set')
        X_train.to_pickle("X_train.pickle")
        X_test.to_pickle("X_test.pickle")

    except Exception as e:
        logger.error(e)