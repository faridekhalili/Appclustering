from preprocessor import pre_process
import pandas as pd
import pickle
import toml
import argparse
import re
from ast import literal_eval
from gensim.models.ldamulticore import LdaMulticore


def get_dominant_topic(new_docs, dictionary, tfidf, topic_model):
    for doc in list(new_docs):
        vec_bow = dictionary.doc2bow(doc)
        vec_tfidf = tfidf[vec_bow]
        if len(topic_model[vec_tfidf]) == 0:
            print("can't find the model")
            return
        else:
            topic_distribution = dict(topic_model[vec_tfidf])
            dominant_topic = max(topic_distribution, key=topic_distribution.get)
    return dominant_topic


def get_required_files_and_models(query_doc, path):
    with open(path + "dictionary", 'rb') as pickle_file:
        dictionary = pickle.load(pickle_file)
    with open(path + "tfidf_model", 'rb') as pickle_file:
        tfidf = pickle.load(pickle_file)
    topic_model = LdaMulticore.load(path + "model")
    with open(query_doc, 'r') as file:
        application_description = file.read().replace('\n', '')
    df = pd.DataFrame(application_description, columns=["description"])
    df["description"] = pre_process(df[['description']])
    preprocessed_application_description = [literal_eval(x) for x in list(df["description"])]
    return dictionary, tfidf, topic_model, preprocessed_application_description


def get_best_word_embedding_model(algorithm, query_doc, path):
    dictionary, tfidf, topic_model, preprocessed_application_description = \
        get_required_files_and_models(query_doc, path)

    dominant_topic = get_dominant_topic(preprocessed_application_description, dictionary,
                                        tfidf, topic_model)

    model_path = path + algorithm + "_models/model_" + str(dominant_topic)
    return model_path


def check_inputs(args):
    flag = False
    if args.algorithm is None:
        args.algorithm = "word2vec"
    if args.file_name is None:
        print("You have to provide the file name(a .txt) where the description of the algorithm of your query resides.")
        flag = True
    if re.search('.txt', args.file_name) is None:
        print("Your file name must be a .txt")
        flag = True
    return flag, args


def main():
    conf = toml.load('config.toml')
    query_input_path = conf['query_input_path']
    query_result_path = conf['query_result_path']
    best_topic_model_path = conf['best_topic_model_path']
    parser = argparse.ArgumentParser(description='word embedding retrieving script')
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='word embedding algorithm')
    parser.add_argument('--file_name', dest='file_name', type=str,
                        help='The file name of where the query resides')
    args = parser.parse_args()
    flag, args = check_inputs(args)
    if flag:
        return
    model_path = get_best_word_embedding_model(args.algorithm, query_input_path +
                                               args.file_name, best_topic_model_path)
    f = open(query_result_path + "word_embedding_path.txt", "w")
    f.write(model_path)
    f.close()


if __name__ == "__main__":
    main()
