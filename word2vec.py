import toml
import time
import pandas as pd
import numpy as np
import warnings
from gensim.models import Word2Vec
from ast import literal_eval
import argparse
from utils import *


def word2vec_trainer(df, model_path, size=70):
    list_of_tokens = list(df["description"])
    if isinstance(list_of_tokens[0], str):
        list_of_tokens = [literal_eval(x) for x in list_of_tokens]
    start_time = time.time()
    model = Word2Vec(list_of_tokens, min_count=1, size=size, workers=3, window=3, sg=1)
    print("Time taken to train the word2vec model: " + str(int((time.time() - start_time) / 60)) + ' minutes\n')
    model.save(model_path)


def extract_word2vec_models(folder_path, algorithm):
    extended_df = pd.read_csv(folder_path + algorithm + '/labeled.csv')
    count = 0
    word2vec_models_path = folder_path + algorithm + '/word2vec_models/'
    start_all_time = time.time()
    for category, df_category in extended_df.groupby('topic'):
        start_time = time.time()
        count += 1
        model_name = word2vec_models_path + str(count) + ".model"
        word2vec_trainer(df=df_category, model_path=model_name)
        write_to_file(
            "Time taken to train the " + str(count) + "th word2vec model resulting from " + str(algorithm) + ": " + str(
                int((time.time() - start_time) / 60)) + ' minutes\n')
    write_to_file("Time taken to train all the word2vec models: " + str(
        int((time.time() - start_all_time) / 60)) + ' minutes\n\n')
    write_to_file(80 * "#" + '\n\n')
    print("extract_word2vec_models")


def main():
    conf = toml.load('config.toml')
    topic_modeling_path = conf['topic_modeling_path']
    parser = argparse.ArgumentParser(description='Topic modeling software')
    parser.add_argument("--modelNumbers", nargs="*")
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='topic modeling algorithm')
    args = parser.parse_args()
    if args.algorithm is None:
        print('set algorithm first')
        return
    if args.modelNumbers is None or args.algorithm is None:
        extract_word2vec_models(topic_modeling_path, "lsa")
        extract_word2vec_models(topic_modeling_path, "lda")
        extract_word2vec_models(topic_modeling_path, "hdp")
    else:
        model_path = topic_modeling_path + args.algorithm
        extended_df = pd.read_csv(model_path + '/labeled.csv')
        for model_number in args.modelNumbers:
            if int(model_number) > extended_df["topic"].max():
                print("Sorry there is no category " + model_number + " created from the " + args.algorithm)
                continue
            start_time = time.time()
            model_name = model_path + "/word2vec_models/" + model_number + ".model"
            df_category = extended_df.loc[extended_df['topic'] == int(model_number)]
            word2vec_trainer(df=df_category, model_path=model_name)
            timing_log = "Time taken to train the " + model_number + \
                         "th word2vec model resulting from " + str(args.algorithm) + ": " + \
                         str(int((time.time() - start_time) / 60)) + ' minutes\n'
            write_to_file(timing_log)


if __name__ == "__main__":
    main()
