from preprocessor import pre_process
import pandas as pd
import pickle
import toml
import argparse
import re
from ast import literal_eval


def get_dominant_topic(new_docs, dictionary, tfidf, topic_model, models_path):
    for doc in list(new_docs):
        vec_bow = dictionary.doc2bow(doc)
        vec_tfidf = tfidf[vec_bow]
        if len(topic_model[vec_tfidf]) == 0:
            print("can't find the model")
            return
        else:
            topic_distribution = dict(topic_model[vec_tfidf])
            dominant_topic = max(topic_distribution, key=topic_distribution.get)
            best_model = models_path + "w2v_model_" + str(dominant_topic)
    return best_model


def get_best_word2vec_model(algorithm, query_doc, path):
    with open(open(path + "dictionary"), 'rb') as pickle_file:
        dictionary = pickle.load(pickle_file)
    with open(open(path + "tfidf_model"), 'rb') as pickle_file:
        tfidf = pickle.load(pickle_file)
    with open(query_doc, 'r') as file:
        application_description = file.read().replace('\n', '')
    df = pd.DataFrame(application_description, columns=["description"])
    df["description"] = pre_process(df[['description']])
    preprocessed_application_description = [literal_eval(x) for x in list(df["description"])]

    if algorithm == "lda":
        model_path = path + 'lda/model/lda.model'
        with open(open(model_path), 'rb') as pickle_file:
            lda_model = pickle.load(pickle_file)
        word2vec_models_path = path + 'lda/word2vec_models/'
        word2vec_model = get_dominant_topic(preprocessed_application_description, dictionary,
                                            tfidf, lda_model, word2vec_models_path)
    elif algorithm == "lsa":
        model_path = path + 'lsa/model/lsa.model'
        with open(open(model_path), 'rb') as pickle_file:
            lsa_model = pickle.load(pickle_file)
        word2vec_models_path = path + 'lsa/word2vec_models/'
        word2vec_model = get_dominant_topic(preprocessed_application_description,
                                                 dictionary, tfidf, lsa_model,
                                                 word2vec_models_path)
    elif algorithm == "hdp":
        model_path = path + 'hdp/model/hdp.model'
        with open(open(model_path), 'rb') as pickle_file:
            hdp_model = pickle.load(pickle_file)
        word2vec_models_path = path + 'hdp/word2vec_models/'
        word2vec_model = get_dominant_topic(preprocessed_application_description,
                                            dictionary, tfidf, hdp_model,
                                            word2vec_models_path)
    return word2vec_model


def check_inputs(args):
    flag = False
    if args.algorithm is None:
        args.algorithm = "lda"
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
    topic_modeling_path = conf['topic_modeling_path']
    parser = argparse.ArgumentParser(description='Word2vec retrieving script')
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='topic modeling algorithm')
    parser.add_argument('--file_name', dest='file_name', type=str,
                        help='The file name of where the query resides')
    args = parser.parse_args()
    flag, args = check_inputs(args)
    if flag:
        return
    model = get_best_word2vec_model(args.algorithm, query_input_path + args.file_name,
                                    topic_modeling_path)
    pickle.dump(model, open(query_result_path + args.file_name[:-4] + "_w2v_model", "wb"))


if __name__ == "__main__":
    main()
