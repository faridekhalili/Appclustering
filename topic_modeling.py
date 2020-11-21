import argparse
import toml
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
from utils import *
from abc import ABC, abstractmethod
from ast import literal_eval


class TopicModel(ABC):
    def __init__(self, dataset, folder_path, algorithm, min_topics, max_topics, step):
        self.num_topics = list(range(min_topics, max_topics, step))
        self.dataset = dataset
        self.folder_path = folder_path
        self.algorithm = algorithm
        self.dictionary, self.corpus_tfidf = load_dictionary_and_tfidf_corpus(dataset, folder_path)
        print("init done")
        super().__init__()

    def __plot_coherence_scores(self, coherence_scores):
        png_name = str(self.num_topics[0]) + "_" + \
                   str(self.num_topics[1] - self.num_topics[0]) + "_" + \
                   str(self.num_topics[-1])
        figure_path = self.folder_path + self.algorithm + '/' + png_name + '_coherence.png'
        save_coherence_plot(self.num_topics, coherence_scores, figure_path)
        print("__plot_coherence_scores")

    def search_num_of_topics(self):
        file_name = self.folder_path + self.algorithm + '/' + \
                    str(self.num_topics[0]) + "_" + \
                    str(self.num_topics[1] - self.num_topics[0]) + "_" + \
                    str(self.num_topics[-1]) + ".csv"
        coherence_scores = []
        for i in self.num_topics:
            print(i)
            model = self.get_model(i)
            cm = CoherenceModel(model=model, texts=self.dataset,
                                corpus=self.corpus_tfidf, coherence='c_v')
            coherence_scores.append(cm.get_coherence())
        coherence_scores_df = pd.DataFrame(
            {'num_topics': self.num_topics,
             'coherence_scores': coherence_scores,
             })
        coherence_scores_df.to_csv(file_name)
        self.__plot_coherence_scores(coherence_scores)
        print("search_num_of_topics")

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

    def __init__(self, dataset, folder_path, algorithm, min_topics=1, max_topics=101, step=10):
        super().__init__(dataset, folder_path, algorithm, min_topics, max_topics, step)

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

    def __init__(self, dataset, folder_path, algorithm, min_topics=1, max_topics=101, step=10):
        super().__init__(dataset, folder_path, algorithm, min_topics, max_topics, step)

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

    def __init__(self, dataset, folder_path, algorithm, min_topics=1, max_topics=101, step=10):
        super().__init__(dataset, folder_path, algorithm, min_topics, max_topics, step)

    def get_model(self):
        start_time = time.time()
        hdp_model = gensim.models.hdpmodel.HdpModel(corpus=self.corpus_tfidf, id2word=self.dictionary)
        write_to_file('\n\n' + str(hdp_model.print_topics(num_words=10)) + '\n\n')
        pprint(hdp_model.print_topics(num_words=10))
        print("training time of HDP model: " + str(int((time.time() - start_time) / 60)) + ' minutes\n')
        write_to_file("Time taken to train the hdp model: " + str(int((time.time() - start_time) / 60)) + ' minutes\n')
        model_path = self.folder_path + self.algorithm + '/model/' + self.algorithm + '.model'
        hdp_model.save(model_path)
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


def main():
    parser = argparse.ArgumentParser(description='Topic modeling software')
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='topic modeling algorithm')
    parser.add_argument('--min', dest='min_topics', type=int, help='min number of topics')
    parser.add_argument('--max', dest='max_topics', type=int, help='max number of topics')
    parser.add_argument('--step', dest='step_topics', type=int, help='step to increment')
    args = parser.parse_args()

    conf = toml.load('config.toml')
    topic_modeling_path = conf['topic_modeling_path']
    print("reading df")
    df = pd.read_csv(conf["preprocessed_data_path"])
    empty_description_indices = df[df["description"] == '[]'].index
    print(len(df))
    df.drop(empty_description_indices, inplace=True)
    print(len(df))
    print("df read")
    texts = [literal_eval(x) for x in list(df["description"])]
    print("texts created")
    del df

    if args.algorithm == "lsa":
        lsa_obj = LSA(texts, topic_modeling_path, "lsa", args.min_topics, args.max_topics, args.step_topics)
        del texts

        lsa_obj.search_num_of_topics()
        del lsa_obj

    elif args.algorithm == "lsa":
        lda_obj = LDA(texts, topic_modeling_path, "lda",
                      args.min_topics, args.max_topics, args.step_topics)
        del texts
        lda_obj.search_num_of_topics()
        del lda_obj

    elif args.algorithm == "hdp":
        hdp_obj = HDP(texts, topic_modeling_path, "hdp",
                      args.min_topics, args.max_topics, args.step_topics)
        del texts
        hdp_model = hdp_obj.get_model()
        hdp_obj.topic_prob_extractor(hdp_model)
        del hdp_obj


if __name__ == "__main__":
    main()
