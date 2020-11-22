import pandas as pd
from ast import literal_eval
import toml
import argparse
import glob

from topic_modeling import save_coherence_plot
from utils import *
from pprint import pprint
import matplotlib.pyplot as plt


def save_topic_model(model, folder_path, algorithm):
    model_path = folder_path + algorithm + '/model/' + algorithm + '.model'
    pickle.dump(model, open(model_path, 'wb'))
    print("save_topic_model")


def extract_dominant_topics(model, dataset, folder_path):
    topic_clusters = []
    dictionary, corpus_tfidf = load_dictionary_and_tfidf_corpus(dataset, folder_path)
    for i in range(len(corpus_tfidf)):
        topic_distribution = dict(model[corpus_tfidf[i]])
        dominant_topic = max(topic_distribution, key=topic_distribution.get)
        topic_clusters.append(dominant_topic)
    print("__extract_dominant_topics")
    return topic_clusters


def divide_into_clusters(model, df, folder_path, algorithm):
    texts = [literal_eval(x) for x in list(df["description"])]
    topic_clusters = extract_dominant_topics(model, texts, folder_path)
    extended_df = pd.DataFrame(list(zip(texts, topic_clusters, list(df["category"]))),
                               columns=['description', 'topic', 'category'])
    extended_df.to_csv(folder_path + algorithm + '/labeled.csv')
    print("divide_into_clusters")


def get_best_topic_model(df, folder_path, algorithm):
    texts = [literal_eval(x) for x in list(df["description"])]
    dictionary, corpus_tfidf = load_dictionary_and_tfidf_corpus(texts, folder_path)
    best_number_topics = get_optimal_number_from_cv(algorithm, folder_path)
    print("best_model retrieved: " + str(best_number_topics))
    if algorithm == "lda":
        model = gensim.models.LdaMulticore(corpus_tfidf,
                                           num_topics=best_number_topics,
                                           id2word=dictionary,
                                           passes=4, workers=10, iterations=100)
    elif algorithm == "lsa":
        model = gensim.models.LsiModel(corpus_tfidf,
                                       num_topics=best_number_topics,
                                       id2word=dictionary)
    return model


def get_optimal_number_from_cv(algorithm, folder_path):
    path = folder_path + algorithm
    all_files = glob.glob(path + "/*.csv")  # todo here is a bug. it also considers labeled
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    best_number_topics = df.iloc[df['coherence_scores'].argmax()]["num_topics"]
    df.sort_values(by=['num_topics'], inplace=True)
    save_coherence_plot(list(df['num_topics']), list(df['coherence_scores']), path+'/overall_cv.png')
    return best_number_topics


def plot_distribution(df, plot_path, col):
    plt.figure(figsize=(15, 5))
    pd.value_counts(df[col]).plot.bar(title="category distribution in the dataset")
    plt.xlabel("Topic")
    plt.ylabel("Number of applications in the topic")
    plt.savefig(plot_path)


def main():
    parser = argparse.ArgumentParser(description='Topic modeling software')
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='topic modeling algorithm')
    args = parser.parse_args()

    conf = toml.load('config.toml')
    topic_modeling_path = conf['topic_modeling_path']

    df = pd.read_csv(conf["preprocessed_data_path"])

    if args.algorithm == "hdp":
        model_path = topic_modeling_path + 'hdp/model/hdp.model'
        hdp_model = pickle.load(open(model_path, 'rb'))
        divide_into_clusters(hdp_model, df, topic_modeling_path, args.algorithm)
    else:
        model = get_best_topic_model(df, topic_modeling_path, args.algorithm)
        write_to_file(args.algorithm + " topics : \n\n")
        write_to_file('\n\n' + str(model.print_topics()) + '\n\n')
        pprint(model.print_topics())
        save_topic_model(model, topic_modeling_path, args.algorithm)
        divide_into_clusters(model, df, topic_modeling_path, args.algorithm)

    distribution_plot_path = topic_modeling_path + args.algorithm + '/topic_distribution.png'
    extended_df = pd.read_csv(topic_modeling_path + args.algorithm + '/labeled.csv')
    plot_distribution(extended_df, distribution_plot_path, 'topic')


if __name__ == "__main__":
    main()
