import argparse
import pickle
import gensim


def pickle_save(my_model, file_name):
    pickle.dump(my_model, open(file_name, 'wb'))


def pickle_load(file_name):
    loaded_obj = pickle.load(open(file_name, 'rb'))
    return loaded_obj


def write_to_file(message):
    f = open('./output/topic_modeling/timings.txt', 'a')
    f.write(message)



def load_dictionary_and_tfidf_corpus(dataset, folder_path): # todo here data set have loaded and it will not be used
    # if corpus is already save. So there is extra un necessary operation.
    dictionary_path = folder_path + "dictionary"
    tfidf_path = folder_path + "tfidf_model"
    tfidf_corpus_path = folder_path + "tfidf_corpus"
    try:
        dictionary = pickle.load(open(dictionary_path, "rb"))
        corpus_tfidf = pickle.load(open(tfidf_corpus_path, "rb"))
    except (OSError, IOError) as e:
        dictionary = gensim.corpora.Dictionary(dataset)
        dictionary.filter_extremes(no_below=20, no_above=0.13, keep_n=len(dictionary))
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
    parser.add_argument('--min', dest='min_topics', type=int, help='min number of topics')
    parser.add_argument('--max', dest='max_topics', type=int, help='max number of topics')
    parser.add_argument('--step', dest='step_topics', type=int, help='step to increment')
    args = parser.parse_args()
    return args
