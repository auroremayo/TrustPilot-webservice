import requests
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# Regrouper les notes
def regrouper_notes(y):
    """
    Regroupe les notes :
    - 1, 2 -> 1 (négatif)
    - 3    -> 2 (neutre)
    - 4, 5 -> 3 (positif)
    """
    return y.replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 3})


def equilibrer_par_index(x_df, y_series_grouped):
    # Obtenir les index par classe
    classes = y_series_grouped.unique()
    series_by_class = [y_series_grouped[y_series_grouped == c] for c in classes]
    
    # Taille minimale pour équilibrer
    min_count = min(len(s) for s in series_by_class)
    
    # Sous-échantillonnage équilibré
    sampled_idx = [s.sample(n=min_count, random_state=42).index for s in series_by_class]
    final_idx = pd.Index(np.concatenate(sampled_idx)).sort_values()
    
    # Appliquer aux X et Y
    x_balanced = x_df.loc[final_idx]
    y_balanced = y_series_grouped.loc[final_idx]
    
    return x_balanced, y_balanced



                
def main():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    
    logger.info('making raw data set')
    df = pd.read_csv("data/raw_data/df_merged_clean.csv", sep=",")
    #df = df.head(1000)  # Example: keep only the first 1000 rows
    n_classes = df['overall'].nunique()
    n_par_classe = 1000 // n_classes

    sample = (
        df.groupby('overall', group_keys=False)
        .apply(lambda x: x.sample(n=n_par_classe, random_state=42), include_groups=42)
    )

    df = sample.reset_index(drop=True)

    df = df.dropna(subset=['summary', 'reviewText', 'overall'])
    df = df.rename(columns={'overall': 'score'})
    df['score'] = df['score'].astype(int)


    target = df['score']
    feats = df[['summary', 'reviewText']]

    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state = 42)

    y_train_grouped = regrouper_notes(y_train)
    y_test_grouped = regrouper_notes(y_test)

    x_train_bal, y_train_bal = equilibrer_par_index(X_train, y_train_grouped)
    x_test_bal, y_test_bal = equilibrer_par_index(X_test, y_test_grouped)

    y_train_bal = y_train_bal - 1
    y_test_bal = y_test_bal - 1

    print("Train équilibré :", y_train_bal.value_counts())
    print("Test équilibré :", y_test_bal.value_counts())

    for file, filename in zip([x_train_bal, x_test_bal, y_train_bal, y_test_bal], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join("data/processed_data", f'{filename}.csv')
        file.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()