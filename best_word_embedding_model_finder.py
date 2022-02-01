from preprocessor import pre_process
import pandas as pd
import pickle
import toml
import argparse
from ast import literal_eval
from gensim.models.ldamulticore import LdaMulticore
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError
from play_scraper import details
import time
import os
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


def get_dominant_topic(new_docs, dictionary, tfidf, topic_model):
    for doc in list(new_docs):
        vec_bow = dictionary.doc2bow(doc)
        vec_tfidf = tfidf[vec_bow]pickle.load
        if len(topic_model[vec_tfidf]) == 0:
            print("can't find the model")
            return
        else:
            topic_distribution = dict(topic_model[vec_tfidf])
            dominant_topic = max(topic_distribution, key=topic_distribution.get)
    return dominant_topic


def get_preprocessed_desc(desc):
    df = pd.DataFrame([desc], columns=["description"])
    df["description"] = pre_process(df[['description']])
    # preprocessed_application_description = [literal_eval(x) for x in list(df["description"])]
    tokenized_data = df[['description']].applymap(lambda s: word_tokenize(s))
    preprocessed_application_description = list(tokenized_data["description"])
    return preprocessed_application_description


def get_required_models(algorithm, best_topic_model_path, topic_modeling_path):
    with open(topic_modeling_path + "dictionary", 'rb') as pickle_file:
        dictionary = pickle.load(pickle_file)
    with open(topic_modeling_path + "tfidf_model", 'rb') as pickle_file:
        tfidf = pickle.load(pickle_file)
    topic_model = LdaMulticore.load(best_topic_model_path + algorithm + "/model/"+algorithm+".model")
    return dictionary, tfidf, topic_model


def get_best_word_embedding_model(args, desc, best_topic_model_path, topic_modeling_path):
    dictionary, tfidf, topic_model = get_required_models(args.algorithm, best_topic_model_path, topic_modeling_path)
    preprocessed_application_description = get_preprocessed_desc(desc)
    dominant_topic = get_dominant_topic(preprocessed_application_description, dictionary,
                                        tfidf, topic_model)
    if args.word_embedding == "glove":
        model_path = best_topic_model_path + args.algorithm + "/glove_models/word2vec_format/model_" +\
                     str(dominant_topic)+".txt"
    else:
        model_path = best_topic_model_path + args.algorithm + "/" + args.word_embedding +"/model_" + str(dominant_topic)
    return model_path


def app_details(app_id):
    for i in range(3):
        try:
            return details(app_id)
        except (ReadTimeout, ConnectionError):
            print("ReadTimeout error, waiting for "+str(i ** 3)+ "seconds.")
        except (HTTPError, ValueError):
            print("url for " + str(app_id) + "not found")
            return
        except AttributeError:
            print("AttributeError")
        time.sleep(i ** 3)


def get_description(app_name_to_id_path, app_name):
    df = pd.read_csv(app_name_to_id_path)
    tdf = df[df["app_name"] == app_name]
    if len(tdf) != 1:
        print(str(len(df)) + "application ids found for this application name while it should have been 1.")
        return
    details = app_details(list(tdf["app_id"])[0])
    if details is None:
        return
    desc = details["description"]
    return desc


def check_inputs(args):
    flag = False
    if args.algorithm is None:
        args.algorithm = "lda"
    if args.word_embedding is None:
        args.word_embedding = "w2v"
    if args.app_name is None:
        print("You have to provide the name of the application you want to find a word embedding model for.")
        flag = True
    return flag, args


def main():
    conf = toml.load('config.toml')
    app_name_to_id_path = conf['app_name_to_id_path']
    query_result_path = conf['query_result_path']
    best_topic_model_path = conf['best_topic_model_path']
    topic_modeling_path = conf['topic_modeling_path']
    parser = argparse.ArgumentParser(description='word embedding retrieving script')
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='Topic modeling algorithm')
    parser.add_argument('--word_embedding', dest='word_embedding', type=str, help='word embedding algorithm')
    parser.add_argument('--app_name', dest='app_name', type=str,
                        help='The name of the application you want to find the word embedding model for')
    args = parser.parse_args()
    flag, args = check_inputs(args)
    if flag:
        return
    desc = get_description(app_name_to_id_path, args.app_name)
    model_path = get_best_word_embedding_model(args, desc, best_topic_model_path, topic_modeling_path)
    if not os.path.exists(query_result_path):
        os.makedirs(query_result_path)
    f = open(query_result_path + args.app_name+
             "_"+args.word_embedding+".txt", "w")
    f.write(model_path)
    f.close()


if __name__ == "__main__":
    main()
