from preprocessor import pre_process
import pandas as pd
import pickle
import toml
import argparse
import re
from ast import literal_eval


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


def get_best_word_embedding_model(algorithm, query_doc, path):
    with open(open(path + "dictionary"), 'rb') as pickle_file:
        dictionary = pickle.load(pickle_file)
    with open(open(path + "tfidf_model"), 'rb') as pickle_file:
        tfidf = pickle.load(pickle_file)
    with open(query_doc, 'r') as file:
        application_description = file.read().replace('\n', '')
    df = pd.DataFrame(application_description, columns=["description"])
    df["description"] = pre_process(df[['description']])
    preprocessed_application_description = [literal_eval(x) for x in list(df["description"])]

    model_path = path + algorithm + "model/" + algorithm + ".model"
    with open(open(model_path), 'rb') as pickle_file:
        lda_model = pickle.load(pickle_file)
    dominant_topic = get_dominant_topic(preprocessed_application_description, dictionary,
                                        tfidf, lda_model)
    w2v_model_path = path + algorithm + '/word2vec_models/' + "w2v_model_" + str(dominant_topic)
    with open(open(w2v_model_path), 'rb') as pickle_file:
        w2v_model = pickle.load(pickle_file)
    fast_text_model_path = path + algorithm + '/word2vec_models/' + "model_" + str(dominant_topic)
    with open(open(fast_text_model_path), 'rb') as pickle_file:
        fast_text_model = pickle.load(pickle_file)
    glove_model_path = path + algorithm + '/word2vec_models/' + "model_" + str(dominant_topic)
    with open(open(glove_model_path), 'rb') as pickle_file:
        glove_model = pickle.load(pickle_file)
    return w2v_model, fast_text_model, glove_model


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
    w2v_model, fast_text_model, glove_model = get_best_word_embedding_model(args.algorithm,
                                                                            query_input_path + args.file_name,
                                                                            topic_modeling_path)
    pickle.dump(w2v_model, open(query_result_path + "word2vec" + args.file_name[:-4] + "_w2v_model", "wb"))
    pickle.dump(fast_text_model, open(query_result_path + "FastText" + args.file_name[:-4] + "_model", "wb"))
    pickle.dump(glove_model, open(query_result_path + "Glove" + args.file_name[:-4] + "_model", "wb"))


if __name__ == "__main__":
    main()
