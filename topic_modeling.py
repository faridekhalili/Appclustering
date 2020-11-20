import toml
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
from utils import *
from abc import ABC, abstractmethod
from ast import literal_eval
import pickle


class TopicModel(ABC):
    def __init__(self, dataset, folder_path, algorithm):
        self.num_topics = list(range(1, 21, 10))
        self.dataset = dataset
        self.folder_path = folder_path
        self.algorithm = algorithm
        self.dictionary, self.corpus_tfidf = load_dictionary_and_tfidf_corpus(dataset, folder_path)
        print("init done")
        super().__init__()

    def __plot_coherence_scores(self, coherence_scores):
        figure_path = self.folder_path + self.algorithm + '/' + self.algorithm + '_coherence.png'
        save_coherence_plot(self.num_topics, coherence_scores, figure_path)
        print("__plot_coherence_scores")

    def __extract_dominant_topics(self, best_model):
        topic_clusters = []
        remove_indices = []
        for i in range(len(self.corpus_tfidf)):
            if len(best_model[self.corpus_tfidf[i]]) == 0:
                remove_indices.append(i)
            else:
                topic_distribution = dict(best_model[self.corpus_tfidf[i]])
                dominant_topic = max(topic_distribution, key=topic_distribution.get)
                topic_clusters.append(dominant_topic)
        self.dataset = [i for j, i in enumerate(self.dataset) if j not in remove_indices]
        print("__extract_dominant_topics")
        return topic_clusters

    def save_topic_model(self, model):
        model_path = self.folder_path + self.algorithm + '/model/' + self.algorithm + '.model'
        model.save(model_path)
        print("save_topic_model")

    def search_num_of_topics(self):
        coherence_scores = []
        for i in self.num_topics:
            print(i)
            model = self.get_model(i)
            cm = CoherenceModel(model=model, texts=self.dataset,
                                corpus=self.corpus_tfidf, coherence='c_v')
            coherence_scores.append(cm.get_coherence())
        best_num_topics = self.num_topics[coherence_scores.index(max(coherence_scores))]
        print("best_num_topics: " + str(best_num_topics))
        best_model = self.get_model(best_num_topics)
        print("best_model retrieved")
        write_to_file('\n\n' + str(best_model.print_topics()) + '\n\n')
        pprint(best_model.print_topics())
        self.save_topic_model(best_model)
        self.__plot_coherence_scores(coherence_scores)
        print("search_num_of_topics")
        return best_model

    def divide_into_clusters(self, best_model):
        topic_clusters = self.__extract_dominant_topics(best_model)
        extended_df = pd.DataFrame(list(zip(self.dataset, topic_clusters)), columns=['description', 'topic'])
        extended_df.to_csv(self.folder_path + self.algorithm + '/labeled.csv')
        print("divide_into_clusters")

    def topic_prob_extractor(self, model):
        shown_topics = model.print_topics(num_topics=150, num_words=500)
        topics_nos = [x[0] for x in shown_topics]
        weights = [sum([float(item.split("*")[0]) for item in shown_topics[topicN][1].split("+")]) for topicN in
                   topics_nos]
        df = pd.DataFrame({'topic_id': topics_nos, 'weight': weights})
        index_names = df[df['weight'] == 0.0].index
        df.drop(index_names, inplace=True)
        topic_wight_df_path = self.folder_path + self.algorithm + '/topic_wight_df.csv'
        df.to_csv(topic_wight_df_path)
        print("topic_prob_extractor")
        return df

    @abstractmethod
    def get_model(self, num_topics):
        pass


class LSA(TopicModel):

    def __init__(self, dataset, folder_path, algorithm):
        super().__init__(dataset, folder_path, algorithm)

    def get_model(self, num_topics):
        start_time = time.time()
        lsa_model = gensim.models.LsiModel(self.corpus_tfidf,
                                           num_topics=num_topics,
                                           id2word=self.dictionary)
        timing_log = "training time of LSA model with " + str(num_topics) + " number of topics: " + str(
            int((time.time() - start_time) / 60)) + ' minutes\n'
        print(timing_log)
        write_to_file(timing_log)
        return lsa_model


class LDA(TopicModel):

    def __init__(self, dataset, folder_path, algorithm):
        super().__init__(dataset, folder_path, algorithm)

    def get_model(self, num_topics):
        start_time = time.time()
        lda_model = gensim.models.LdaMulticore(self.corpus_tfidf,
                                               num_topics=num_topics,
                                               id2word=self.dictionary,
                                               passes=4, workers=10, iterations=100)
        timing_log = "training time of LDA model with " + str(num_topics) + " number of topics: " + str(
            int((time.time() - start_time) / 60)) + ' minutes\n'
        print(timing_log)
        write_to_file(timing_log)
        return lda_model


class HDP(TopicModel):

    def __init__(self, dataset, folder_path, algorithm):
        super().__init__(dataset, folder_path, algorithm)

    def get_model(self):
        start_time = time.time()
        hdp_model = gensim.models.hdpmodel.HdpModel(corpus=self.corpus_tfidf, id2word=self.dictionary)
        write_to_file('\n\n' + str(hdp_model.print_topics(num_words=10)) + '\n\n')
        pprint(hdp_model.print_topics(num_words=10))
        print("training time of HDP model: " + str(int((time.time() - start_time) / 60)) + ' minutes\n')
        write_to_file("Time taken to train the hdp model: " + str(int((time.time() - start_time) / 60)) + ' minutes\n')
        self.save_topic_model(hdp_model)
        return hdp_model


def save_coherence_plot(num_topics, coherence_scores, figure_path):
    plt.figure(figsize=(10, 5))
    plt.plot(num_topics, coherence_scores)
    plt.xticks(np.arange(min(num_topics), max(num_topics) + 1, num_topics[1] - num_topics[0]))
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence score')
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def load_dictionary_and_tfidf_corpus(dataset, folder_path):
    dictionary_path = folder_path + "dataset.dict"
    tfidf_path = folder_path + "dataset.tfidf_model"
    tfidf_corpus_path = folder_path + "tfidf_corpus"
    try:
        dictionary = pickle.load(open(dictionary_path, "rb"))
        corpus_tfidf = pickle.load(open(tfidf_corpus_path, "rb"))
    except (OSError, IOError) as e:
        dictionary = gensim.corpora.Dictionary(dataset)
        pickle.dump(dictionary, open(dictionary_path, "wb"))
        bow_corpus = [dictionary.doc2bow(doc) for doc in dataset]
        tfidf = gensim.models.TfidfModel(bow_corpus)
        pickle.dump(tfidf, open(tfidf_path, "wb"))
        corpus_tfidf = tfidf[bow_corpus]
        pickle.dump(corpus_tfidf, open(tfidf_corpus_path, "wb"))
    return dictionary, corpus_tfidf


def main():
    conf = toml.load('config.toml')
    topic_modeling_path = conf['topic_modeling_path']
    print("reading df")
    df = pd.read_csv(conf["preprocessed_data_path"])
    print("df read")
    texts = [literal_eval(x) for x in list(df["description"])]
    print("texts created")
    del df

    lsa_obj = LSA(texts, topic_modeling_path, "lsa")
    # del texts
    best_lsa_model = lsa_obj.search_num_of_topics()
    lsa_obj.divide_into_clusters(best_lsa_model)
    del lsa_obj

    lda_obj = LDA(texts, topic_modeling_path, "lda")
    # del texts
    best_lda_model = lda_obj.search_num_of_topics()
    lda_obj.divide_into_clusters(best_lda_model)
    del lda_obj

    hdp_obj = HDP(texts, topic_modeling_path, "hdp")
    del texts
    hdp_model = hdp_obj.get_model()
    hdp_obj.topic_prob_extractor(hdp_model)
    hdp_obj.divide_into_clusters(hdp_model)
    del hdp_obj


if __name__ == "__main__":
    main()
