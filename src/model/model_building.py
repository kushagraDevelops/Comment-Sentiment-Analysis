# src/model/model_building.py

import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Logging
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('model_errors.log')

console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(path):
    df = pd.read_csv(path)
    df.fillna('', inplace=True)
    return df


def apply_tfidf(train_text):
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(train_text)

    # Save vectorizer
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    return X, vectorizer


def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        learning_rate=0.05,
        n_estimators=500,
        num_leaves=100,
        max_depth=15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(50)]
    )

    logger.debug("Model training complete")
    return model


def save_model(model):
    with open('lgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)


def main():
    try:
        df = load_data('./data/interim/train_processed.csv')

        X, vectorizer = apply_tfidf(df['clean_comment'])
        y = df['category']

        model = train_model(X, y)

        save_model(model)

        logger.debug("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()