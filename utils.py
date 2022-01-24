import argparse
import pickle
import gensim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


def write_to_file(message):
    f = open('./output/topic_modeling/timings.txt', 'a')
    f.write(message)


def load_dictionary_and_tfidf_corpus(dataset, folder_path):
    dictionary_path = folder_path + "dictionary"
    tfidf_path = folder_path + "tfidf_model"
    tfidf_corpus_path = folder_path + "tfidf_corpus"
    try:
        with open(dictionary_path, 'rb') as pickle_file:
            dictionary = pickle.load(pickle_file)
        with open(tfidf_corpus_path, 'rb') as pickle_file:
            corpus_tfidf = pickle.load(pickle_file)
    except (OSError, IOError) as e:
        dictionary = gensim.corpora.Dictionary(dataset)
        pickle.dump(dictionary, open(dictionary_path, "wb"))
        bow_corpus = [dictionary.doc2bow(doc) for doc in dataset]
        tfidf = gensim.models.TfidfModel(bow_corpus)
        pickle.dump(tfidf, open(tfidf_path, "wb"))
        corpus_tfidf = tfidf[bow_corpus]
        pickle.dump(corpus_tfidf, open(tfidf_corpus_path, "wb"))
    return dictionary, corpus_tfidf


def get_args():
    parser = argparse.ArgumentParser(description='Topic modeling software')
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='topic modeling algorithm')
    parser.add_argument('--word_filter', dest='word_filter', type=str, help='frequency or gaussian')
    parser.add_argument('--document_filter', dest='document_filter', type=str, help='study or top-n or gaussian')
    parser.add_argument('--min', dest='min_topics', type=int, help='min number of topics')
    parser.add_argument('--max', dest='max_topics', type=int, help='max number of topics')
    parser.add_argument('--step', dest='step_topics', type=int, help='step to increment')
    args = parser.parse_args()
    if args.word_filter is None:
        args.word_filter = "frequency"
    if args.document_filter is None:
        args.document_filter = "gaussian"
    return args


def plot_distribution(df, plot_path, col):
    plt.figure(figsize=(15, 5))
    pd.value_counts(df[col]).plot.bar(title="category distribution in the dataset")
    plt.xlabel("Topic")
    plt.ylabel("Number of applications in the topic")
    plt.savefig(plot_path)


def get_guassian_boundary(li, p):
    sigma = np.std(li)
    mu = np.mean(li)
    s = np.random.normal(mu, sigma, 1000000)
    sorted_samples = np.sort(s)
    lower_bound = int(sorted_samples[int(len(sorted_samples) * p / 100)])
    upper_bound = int(sorted_samples[-int(len(sorted_samples) * p / 100)])
    print("retrieved gaussian bounds")
    return lower_bound, upper_bound


def drop_extra_columns(df):
    dropping_columns = []
    for col in df.columns:
        if re.search("^Unnamed*", col) is not None:
            dropping_columns.append(col)
    df.drop(dropping_columns, axis=1, inplace=True)
    return df


def filter_words(df, texts, word_filter):
    dictionary = gensim.corpora.Dictionary(texts)
    filtered_words = set()
    if word_filter == "frequency":
        for k, v in dictionary.dfs.items():
            if v < 0.005 * len(texts) or v > 0.15 * len(texts):
                filtered_words.add(dictionary[k])
    elif word_filter == "gaussian":
        word_frequencies = [v for k, v in dictionary.dfs.items()]
        l, u = get_guassian_boundary(word_frequencies, 47)
        for k, v in dictionary.dfs.items():
            if v < l or v > u:
                filtered_words.add(dictionary[k])
    for i in range(len(texts)):
        texts[i] = [word for word in texts[i] if word not in filtered_words]
    df["description"] = texts
    df['len'] = [len(x) for x in texts]
    df = df[df['len'] > 0]
    print("filter_words done successfully!")
    return df


def filter_documents(df, doc_filter):
    lower_bound = 0
    upper_bound = max(list(df['len']))
    if doc_filter == "study":
        lower_bound = 50
        upper_bound = 1000
    if doc_filter == "top-n":
        df = df.sort_values(by=['len'])
        lower_bound = df.iloc[int(len(df)*5/100)]["len"]
        upper_bound = df.iloc[int(len(df)*95/100)]["len"]
    elif doc_filter == "gaussian":
        lower_bound, upper_bound = get_guassian_boundary(list(df['len']), 17)

    df = df[df['len'] > lower_bound]
    df = df[df['len'] < upper_bound]

    print("filter_documents done successfully")
    return df


def prune_dataset(df, word_filter, doc_filter):
    tokenized_data = df[['description']].applymap(lambda s: word_tokenize(s))
    texts = list(tokenized_data["description"])
    df = drop_extra_columns(df)
    df = filter_words(df, texts, word_filter)
    df = filter_documents(df, doc_filter)
    df.to_csv("./output/D_" + doc_filter + "_W_" + word_filter + ".csv")
    return df
