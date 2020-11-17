from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LsiModel
from gensim.models import Word2Vec
from preprocessor import pre_process
import pandas as pd
import pickle
import toml


def retrieve_w2v_model(model_path):
    if model_path == "none":
        print("sorry no model was found!")
        return
    else:
        test_w2v_model = Word2Vec.load(model_path)
        return test_w2v_model


def pickle_load(file_name):
    loaded_model = pickle.load(open(file_name, 'rb'))
    return loaded_model


def write_result(some_list, filename):
    with open(filename, 'w') as f:
        for item in some_list:
            f.write("%s\n" % item)


def get_dominant_topic(new_docs, dictionary, tfidf, topic_model, models_path):
    word2vec_model_list = []
    for doc in list(new_docs):
        vec_bow = dictionary.doc2bow(doc)
        vec_tfidf = tfidf[vec_bow]
        if len(topic_model[vec_tfidf]) == 0:
            word2vec_model_list.append("none")
        else:
            topic_distribution = dict(topic_model[vec_tfidf])
            dominant_topic = max(topic_distribution, key=topic_distribution.get)
            best_model = models_path + str(dominant_topic) + ".model"
            word2vec_model_list.append(best_model)
    return word2vec_model_list


def get_best_word2vec_model(algorithm, new_docs, path):
    dictionary = pickle_load(path + "dataset.dict")
    tfidf = pickle_load(path + "dataset.tfidf_model")

    df = pd.DataFrame(new_docs, columns=["description"])
    df["description"] = pre_process(df[['description']])
    new_preprocessed_docs = list(df["description"])

    if algorithm == "lda":
        model_path = path + 'lda/model/lda.model'
        lda_model = LdaMulticore.load(model_path)
        word2vec_models_path = path + 'lda/word2vec_models/'
        word2vec_model_list = get_dominant_topic(new_preprocessed_docs, dictionary, tfidf, lda_model,
                                                 word2vec_models_path)
    elif algorithm == "lsa":
        model_path = path + 'lsa/model/lsa.model'
        lsa_model = LsiModel.load(model_path)
        word2vec_models_path = path + 'lsa/word2vec_models/'
        word2vec_model_list = get_dominant_topic(new_preprocessed_docs,
                                                 dictionary, tfidf, lsa_model,
                                                 word2vec_models_path)
    elif algorithm == "hdp":
        model_path = path + 'hdp/model/hdp.model'
        hdp_model = pickle_load(model_path)
        word2vec_models_path = path + 'hdp/word2vec_models/'
        word2vec_model_list = get_dominant_topic(new_preprocessed_docs,
                                                 dictionary, tfidf, hdp_model,
                                                 word2vec_models_path)
    return word2vec_model_list


def main():
    conf = toml.load('config.toml')
    test_input_path = conf['test_input_path']
    test_result_path = conf['test_result_path']
    topic_modeling_path = conf['topic_modeling_path']
    test_df = pd.read_csv(test_input_path)

    lsa_word2vec_model_list = get_best_word2vec_model('lsa', test_df, topic_modeling_path)
    write_result(lsa_word2vec_model_list, test_result_path + 'lsa_results.txt')

    lda_word2vec_model_list = get_best_word2vec_model('lda', test_df, topic_modeling_path)
    write_result(lda_word2vec_model_list, test_result_path + 'lda_results.txt')

    # an example of how we can retrieve the word2vec model of a given test data (say test_df[0])
    test_w2v_model= retrieve_w2v_model(lsa_word2vec_model_list[0])


if __name__ == "__main__":
    main()
