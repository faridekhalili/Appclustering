import ssl

import toml
import re
import string
import pandas as pd
import sqlite3
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def remove_punctuation(s):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return s.translate(translator)


def remove_stop_words(input_str):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(input_str)
    result = [i for i in tokens if i not in stop_words]
    joined_result = ' '.join(result)
    if joined_result == '':
        return input_str
    else:
        return joined_result


def lemmatizing(input_str):
    lemmatizer = WordNetLemmatizer()
    input_str = word_tokenize(input_str)
    result = [lemmatizer.lemmatize(i) for i in input_str]
    return ' '.join(result)


def remove_unusual_char(input_str):
    return re.sub('[^A-Za-z0-9 ]+', '', input_str)


def pre_process(data):
    lower_data = data.applymap(lambda s: s.lower())
    removed_number_data = lower_data.applymap(lambda s: re.sub(r'\d+', '', s))
    removed_punctuation_data = removed_number_data.applymap(lambda s: remove_punctuation(s))
    striped_data = removed_punctuation_data.applymap(lambda s: s.strip())
    removed_extra_white_space = striped_data.applymap(lambda s: ' '.join(s.split()))
    removed_stop_words = removed_extra_white_space.applymap(lambda s: remove_stop_words(s))
    lemmatized_data = removed_stop_words.applymap(lambda s: lemmatizing(s))
    removed_unusual_char = lemmatized_data.applymap(lambda s: remove_unusual_char(s))
    tokenized_data = removed_unusual_char.applymap(lambda s: word_tokenize(s))
    return tokenized_data


def main():
    conf = toml.load('../config-temp.toml')
    # Read sqlite query results into a pandas DataFrame
    con = sqlite3.connect('../' + conf['database_path'])
    df = pd.read_sql_query("SELECT * from app", con)
    con.close()
    df = df.loc[:, ['description', 'category']]
    df["description"] = pre_process(df[['description']])
    df.to_csv('../' + conf['preprocessed_data_path'])


if __name__ == "__main__":
    main()
