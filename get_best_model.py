import gensim
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LsiModel
from Categorization.preprocessor import pre_process
import pandas as pd


def get_dominant_topic(new_docs, dictionary, tfidf, topic_model, models_path) -> list:
    word2vec_model_list = []
    for doc in list(new_docs):
        vec_bow = dictionary.doc2bow(doc)
        vec_tfidf = tfidf[vec_bow]
        topic_distribution = dict(topic_model[vec_tfidf])
        dominant_topic = max(topic_distribution, key=topic_distribution.get)
        best_model = models_path + dominant_topic + ".model"
        word2vec_model_list.append(best_model)
    return word2vec_model_list


def get_best_word2vec_model(algorithm: str, new_docs: list, path: str) -> list:
    tfidf = gensim.models.TfidfModel.load(path + "dataset.tfidf_model")
    dictionary = gensim.corpora.dictionary.load(path + "dataset.dictionary")

    df = pd.DataFrame(new_docs)
    new_docs = pre_process(df[['description']])

    if algorithm == "lda":
        model_path = path + 'lda/model/LDA.model'
        word2vec_models_path = path + 'lda/word2vec_models/'
        lda_model = LdaMulticore.load(model_path)
        word2vec_model_list = get_dominant_topic(new_docs, dictionary, tfidf, lda_model,
                                                 word2vec_models_path)
    elif algorithm == "lsa":
        model_path = path + 'lsa/model/LDA.model'
        word2vec_models_path = path + 'lsa/word2vec_models/'
        lsa_model = LsiModel.load(model_path)
        word2vec_model_list = get_dominant_topic(new_docs,
                                                 dictionary, tfidf, lsa_model,
                                                 word2vec_models_path)
    return word2vec_model_list
