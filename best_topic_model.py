import pandas as pd
from ast import literal_eval
import toml
import argparse
import glob
from utils import *
from pprint import pprint


def save_topic_model(model, folder_path, algorithm):
    model_path = folder_path + algorithm + '/model/' + algorithm + '.model'
    model.save(model_path)
    print("save_topic_model")


def extract_dominant_topics(model, dataset, folder_path):
    topic_clusters = []
    remove_indices = []
    dictionary, corpus_tfidf = load_dictionary_and_tfidf_corpus(dataset, folder_path)
    for i in range(len(corpus_tfidf)):
        if len(model[corpus_tfidf[i]]) == 0:
            remove_indices.append(i)
        else:
            topic_distribution = dict(model[corpus_tfidf[i]])
            dominant_topic = max(topic_distribution, key=topic_distribution.get)
            topic_clusters.append(dominant_topic)
    dataset = [i for j, i in enumerate(dataset) if j not in remove_indices]
    print("__extract_dominant_topics")
    return topic_clusters


def divide_into_clusters(model, dataset, folder_path, algorithm):
    topic_clusters = extract_dominant_topics(model, dataset, folder_path)
    extended_df = pd.DataFrame(list(zip(dataset, topic_clusters)), columns=['description', 'topic'])
    extended_df.to_csv(folder_path + algorithm + '/labeled.csv')
    print("divide_into_clusters")


def get_best_topic_model(dataset, folder_path, algorithm):
    dictionary, corpus_tfidf = load_dictionary_and_tfidf_corpus(dataset, folder_path)
    path = folder_path + algorithm
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.drop(columns=['Unnamed: 0'])
    best_number_topics = frame.iloc[frame['coherence_scores'].argmax()]["num_topics"]
    print("best_model retrieved")
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


def main():
    parser = argparse.ArgumentParser(description='Topic modeling software')
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='topic modeling algorithm')
    args = parser.parse_args()

    conf = toml.load('config.toml')
    topic_modeling_path = conf['topic_modeling_path']
    print("reading df")
    df = pd.read_csv(conf["preprocessed_data_path"])
    print("df read")
    texts = [literal_eval(x) for x in list(df["description"])]
    print("texts created")
    del df

    if args.algorithm == "hdp":
        model_path = topic_modeling_path + 'hdp/model/hdp.model'
        hdp_model = pickle.load(open(model_path, 'rb'))
        divide_into_clusters(hdp_model, texts, topic_modeling_path, args.algorithm)
    else:
        model = get_best_topic_model(texts, topic_modeling_path, args.algorithm)
        write_to_file(args.algorithm+" topics : \n\n")
        write_to_file('\n\n' + str(model.print_topics()) + '\n\n')
        pprint(model.print_topics())
        save_topic_model(model, topic_modeling_path, args.algorithm)
        divide_into_clusters(model, texts, topic_modeling_path, args.algorithm)


if __name__ == "__main__":
    main()
