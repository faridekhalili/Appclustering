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


def app_details(app_id: str) -> dict:
    for i in range(3):
        try:
            return details(app_id)
        except (ReadTimeout, ConnectionError):
            print(f"ReadTimeout error, waiting for {str(i ** 3)} seconds.")
        except (HTTPError, ValueError):
            print("url for %s not found" % app_id)
            return
        except AttributeError:
            print("AttributeError")
        time.sleep(i ** 3)


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


def get_required_models(path):
    with open(path + "dictionary", 'rb') as pickle_file:
        dictionary = pickle.load(pickle_file)
    with open(path + "tfidf_model", 'rb') as pickle_file:
        tfidf = pickle.load(pickle_file)
    topic_model = LdaMulticore.load(path + "model")
    return dictionary, tfidf, topic_model


def get_preprocessed_desc(desc):
    df = pd.DataFrame(desc, columns=["description"])
    df["description"] = pre_process(df[['description']])
    preprocessed_application_description = [literal_eval(x) for x in list(df["description"])]
    return preprocessed_application_description


def get_best_word_embedding_model(algorithm, desc, path):
    dictionary, tfidf, topic_model = get_required_models(path)
    preprocessed_application_description = get_preprocessed_desc(desc)
    dominant_topic = get_dominant_topic(preprocessed_application_description, dictionary,
                                        tfidf, topic_model)
    model_path = path + algorithm + "_models/model_" + str(dominant_topic)
    return model_path


def check_inputs(args):
    flag = False
    if args.algorithm is None:
        args.algorithm = "word2vec"
    if args.app_name is None:
        print("You have to provide the name of the application you want to find a word embedding model for.")
        flag = True
    return flag, args


def get_description(best_topic_model_path, app_name):
    df = pd.read_csv(best_topic_model_path)
    tdf = df[df["app_name"] == app_name]
    if len(tdf) != 1:
        print(str(len(df)) + "application ids found for this application name while it should have been 1.")
        return
    details = app_details(list(tdf["app_id"])[0])
    if details is None:
        return
    desc = details["description"]
    return desc


def main():
    conf = toml.load('config.toml')
    app_name_to_id_path = conf['app_name_to_id_path']
    query_result_path = conf['query_result_path']
    best_topic_model_path = conf['best_topic_model_path']
    parser = argparse.ArgumentParser(description='word embedding retrieving script')
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='word embedding algorithm')
    parser.add_argument('--app_name', dest='app_name', type=str,
                        help='The name of the application you want to find the word embedding model for')
    args = parser.parse_args()
    flag, args = check_inputs(args)
    if flag:
        return
    desc = get_description(app_name_to_id_path, args.app_name)
    model_path = get_best_word_embedding_model(args.algorithm, desc,
                                               best_topic_model_path)
    f = open(query_result_path + "word_embedding_path.txt", "w")
    f.write(model_path)
    f.close()


if __name__ == "__main__":
    main()
