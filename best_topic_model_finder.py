from ast import literal_eval
import toml
import glob
from topic_modeling import save_coherence_plot
from utils import *
from pprint import pprint


def save_topic_model(model, folder_path, algorithm):
    model_path = folder_path + algorithm + '/model/' + algorithm + '.model'
    pickle.dump(model, open(model_path, 'wb'))
    print("save_topic_model")


def extract_dominant_topics(model, df, folder_path):
    texts = [literal_eval(x) for x in list(df["description"])]
    topic_clusters = []
    remove_indices = []
    dictionary, corpus_tfidf = load_dictionary_and_tfidf_corpus(texts, folder_path)
    for i in range(len(corpus_tfidf)):
        if len(model[corpus_tfidf[i]]) == 0:
            remove_indices.append(i)
        else:
            topic_distribution = dict(model[corpus_tfidf[i]])
            dominant_topic = max(topic_distribution, key=topic_distribution.get)
            topic_clusters.append(dominant_topic)
    print("empty model[corpus_tfidf[i]: " + str(remove_indices))
    write_to_file('\n\n' + "empty model[corpus_tfidf[i]]: " + str(remove_indices) + '\n\n')
    print("__extract_dominant_topics")
    return topic_clusters, remove_indices


def divide_into_clusters(model, df, folder_path, algorithm):
    topic_clusters, remove_indices = extract_dominant_topics(model, df, folder_path)
    df.drop(remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)
    extended_df = pd.DataFrame(list(zip(list(df["description"]), topic_clusters, list(df["category"]), list(df["app_id"]))),
                               columns=['description', 'topic', 'category', 'app_id'])
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
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        if filename != path + "/labeled.csv":
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    best_number_topics = df.iloc[df['coherence_scores'].argmax()]["num_topics"]
    df.sort_values(by=['num_topics'], inplace=True)
    save_coherence_plot(list(df['num_topics']), list(df['coherence_scores']), path + '/overall_cv.png')
    return best_number_topics


def main():
    parser = argparse.ArgumentParser(description='Topic modeling software')
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='topic modeling algorithm')
    args = parser.parse_args()

    conf = toml.load('config.toml')
    topic_modeling_path = conf['topic_modeling_path']

    df = pd.read_csv(conf["preprocessed_data_path"])

    if args.algorithm == "hdp":
        model_path = topic_modeling_path + 'hdp/model/hdp.model'
        with open(model_path, 'rb') as pickle_file:
            hdp_model = pickle.load(pickle_file)
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
