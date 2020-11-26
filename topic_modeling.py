import argparse
import toml
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
from utils import *
from preprocessor import *
from abc import ABC, abstractmethod
from ast import literal_eval


def get_range_file_name():
    args = get_args()
    return '_'.join([str(args.min_topics), str(args.max_topics), str(args.step_topics)])


class TopicModel(ABC):
    def __init__(self, dataset, folder_path, algorithm, args):
        self.num_topics = list(range(args.min_topics, args.max_topics, args.step_topics))
        self.dataset = dataset
        self.folder_path = folder_path
        self.algorithm = algorithm
        self.dictionary, self.corpus_tfidf = load_dictionary_and_tfidf_corpus(dataset, folder_path)
        print("init done")
        super().__init__()

    def __plot_coherence_scores(self, coherence_scores, coherence_measure):
        if len(self.num_topics) < 2:
            return
        png_name = get_range_file_name()
        figure_path = self.folder_path + self.algorithm + '/' + png_name + '_' + coherence_measure + '.png'
        save_coherence_plot(self.num_topics, coherence_scores, figure_path)
        print("__plot_coherence_scores")

    def create_models(self):
        file_name = self.folder_path + self.algorithm + '/' + get_range_file_name() + ".csv"
        u_mass_list, c_v_list, c_uci_list, c_npmi_list = [], [], [], []
        for i in self.num_topics:
            print(i)
            model = self.get_model(i)
            u_mass_list.append(CoherenceModel(model=model, texts=self.dataset,
                                              corpus=self.corpus_tfidf, coherence='u_mass').get_coherence())
            c_v_list.append(CoherenceModel(model=model, texts=self.dataset,
                                           corpus=self.corpus_tfidf, coherence='c_v').get_coherence())
            c_uci_list.append(CoherenceModel(model=model, texts=self.dataset,
                                             corpus=self.corpus_tfidf, coherence='c_uci').get_coherence())
            c_npmi_list.append(CoherenceModel(model=model, texts=self.dataset,
                                              corpus=self.corpus_tfidf, coherence='c_npmi').get_coherence())
        coherence_scores_df = pd.DataFrame(
            {'num_topics': self.num_topics,
             'u_mass': u_mass_list,
             'c_v': c_v_list,
             'c_uci': c_uci_list,
             'c_npmi': c_npmi_list,
             })
        coherence_scores_df.to_csv(file_name)
        self.__plot_coherence_scores(u_mass_list, "u_mass")
        self.__plot_coherence_scores(c_v_list, "c_v")
        self.__plot_coherence_scores(c_uci, "c_uci")
        self.__plot_coherence_scores(c_npmi, "c_npmi")
        print("models created")

    @abstractmethod
    def get_model(self, num_topics):
        pass


class LSA(TopicModel):

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


class HDP():

    def __init__(self, dataset, folder_path, algorithm):
        self.dataset = dataset
        self.folder_path = folder_path
        self.algorithm = algorithm
        self.dictionary, self.corpus_tfidf = load_dictionary_and_tfidf_corpus(dataset, folder_path)
        print("init done")
        super().__init__()

    def get_model(self):
        start_time = time.time()
        model_path = self.folder_path + self.algorithm + '/model/' + self.algorithm + '.model'
        hdp_model = gensim.models.hdpmodel.HdpModel(corpus=self.corpus_tfidf, id2word=self.dictionary)
        write_to_file('\n\n' + str(hdp_model.print_topics(num_words=10)) + '\n\n')
        pprint(hdp_model.print_topics(num_words=10))
        print("training time of HDP model: " + str(int((time.time() - start_time) / 60)) + ' minutes\n')
        write_to_file("Time taken to train the hdp model: " + str(int((time.time() - start_time) / 60)) + ' minutes\n')
        pickle.dump(hdp_model, open(model_path, 'wb'))
        return hdp_model

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


def save_coherence_plot(num_topics, coherence_scores, figure_path):
    plt.figure(figsize=(10, 5))
    plt.plot(num_topics, coherence_scores)
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence score')
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def main():
    args = get_args()
    conf = toml.load('config.toml')
    topic_modeling_path = conf['topic_modeling_path']
    print("reading df")
    remove_low_quality_docs(0)
    df = pd.read_csv(conf["preprocessed_data_path"])
    print("df read")
    print(df.columns)
    texts = [literal_eval(x) for x in list(df["description"])]
    print("texts created")
    del df
    # todo use factory method
    if args.algorithm == "lsa":
        lsa_obj = LSA(texts, topic_modeling_path, "lsa", args)
        del texts
        lsa_obj.create_models()
        del lsa_obj

    elif args.algorithm == "lda":
        lda_obj = LDA(texts, topic_modeling_path, "lda", args)
        del texts
        lda_obj.create_models()
        del lda_obj

    elif args.algorithm == "hdp":
        hdp_obj = HDP(texts, topic_modeling_path, "hdp")
        del texts
        hdp_model = hdp_obj.get_model()
        hdp_obj.topic_prob_extractor(hdp_model)
        del hdp_obj


if __name__ == "__main__":
    main()
