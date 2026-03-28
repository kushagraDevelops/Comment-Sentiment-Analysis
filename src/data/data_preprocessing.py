# src/data/data_preprocessing.py

import numpy as np
import pandas as pd
import os
import re
import nltk
import logging
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize once (IMPORTANT)
stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
lemmatizer = WordNetLemmatizer()

# Logging setup
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('preprocessing_errors.log')

console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def preprocess_comment(comment):
    try:
        # Lowercase
        comment = comment.lower().strip()

        # Convert emojis to text (🔥 improvement)
        comment = emoji.demojize(comment)

        # Handle negation (VERY IMPORTANT)
        comment = re.sub(r"not\s+(\w+)", r"not_\1", comment)

        # Remove unwanted characters but keep ! and ?
        comment = re.sub(r'[^a-zA-Z0-9\s!?]', '', comment)

        # Remove stopwords
        words = comment.split()
        words = [word for word in words if word not in stop_words]

        # Lemmatization
        words = [lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)

    except Exception as e:
        logger.error(f"Error: {e}")
        return comment


def normalize_text(df):
    df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
    logger.debug("Text normalization done")
    return df


def save_data(train_data, test_data, data_path):
    interim_path = os.path.join(data_path, 'interim')
    os.makedirs(interim_path, exist_ok=True)

    train_data.to_csv(os.path.join(interim_path, "train_processed.csv"), index=False)
    test_data.to_csv(os.path.join(interim_path, "test_processed.csv"), index=False)

    logger.debug("Processed data saved")


def main():
    try:
        train = pd.read_csv('./data/raw/train.csv')
        test = pd.read_csv('./data/raw/test.csv')

        train = normalize_text(train)
        test = normalize_text(test)

        save_data(train, test, './data')

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()