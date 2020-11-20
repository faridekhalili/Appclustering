import toml
import time
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from ast import literal_eval
import argparse
from utils import *


def plot_distribution(df, plot_path, col):
    plt.figure(figsize=(15, 5))
    pd.value_counts(df[col]).plot.bar(title="category distribution in the dataset")
    plt.xlabel("category")
    plt.ylabel("Number of applications in the dataset")
    plt.savefig(plot_path)


def word2vec_trainer(df, model_path, size=70):
    list_of_tokens = list(df["description"])
    if isinstance(list_of_tokens[0], str):
        list_of_tokens = [literal_eval(x) for x in list_of_tokens]
    start_time = time.time()
    model = Word2Vec(list_of_tokens, min_count=1, size=size, workers=3, window=3, sg=1)
    print("Time taken to train the word2vec model: " + str(int((time.time() - start_time) / 60)) + ' minutes\n')
    model.save(model_path)
    write_w2vec_vectors('.'.join(model_path.split(".")[:-1]) + '_w2v_vectors.csv', df, model, size)


def write_w2vec_vectors(word2vec_filename, df, w2v_model, w2v_vector_size):
    with open(word2vec_filename, 'w+') as word2vec_file:
        for index, row in df.iterrows():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            model_vector = np.nanmean([w2v_model.wv[token] for token in literal_eval(row['description'])],
                                      axis=0).tolist()
            if index == 0:
                header = ",".join(str(ele) for ele in range(w2v_vector_size))
                word2vec_file.write(header)
                word2vec_file.write("\n")
            # Check if the line exists else it is vector of zeros
            if type(model_vector) is list:
                line1 = ",".join([str(vector_element) for vector_element in model_vector])
            else:
                line1 = ",".join([str(0) for i in range(w2v_vector_size)])
            word2vec_file.write(line1)
            word2vec_file.write('\n')


def extract_word2vec_models(folder_path, algorithm):
    distribution_plot_path = folder_path + algorithm + '/topic_distribution.png'
    extended_df = pd.read_csv(folder_path + algorithm + '/labeled.csv')
    plot_distribution(extended_df, distribution_plot_path, 'topic')
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

    if args.modelNumbers is None or args.algorithm is None:
        print("hum")
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
