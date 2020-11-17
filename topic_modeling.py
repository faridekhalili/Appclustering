import gensim
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
from word2vec import *
from abc import ABC, abstractmethod
import pickle


class TopicModel(ABC):
    def __init__(self, dataset, folder_path, algorithm):
        print("init")
        self.max_num_topics = 100
        self.dataset = dataset
        self.folder_path = folder_path
        self.algorithm = algorithm
        self.dictionary = gensim.corpora.Dictionary(self.dataset)
        print("dictionary created")
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.dataset]
        print("bow_corpus created")
        tfidf = gensim.models.TfidfModel(bow_corpus)
        print("tfidf model created")
        self.tfidf = tfidf
        self.corpus_tfidf = tfidf[bow_corpus]
        print("init done")
        super().__init__()

    def __save_dictionary(self):
        dictionary_path = self.folder_path + "dataset.dict"
        pickle_save(self.dictionary, dictionary_path)
        print("__save_dictionary")

    def __save_tfidf_model(self):
        tfidf_path = self.folder_path + "dataset.tfidf_model"
        pickle_save(self.tfidf, tfidf_path)
        print("__save_tfidf_model")

    def __plot_coherence_scores(self, coherence_scores):
        figure_path = self.folder_path + self.algorithm + '/' + self.algorithm + '_coherence.png'
        save_coherence_plot(self.max_num_topics, coherence_scores, figure_path)
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

    def __extract_word2vec_models(self, df):
        distribution_plot_path = self.folder_path + self.algorithm + '/topic_distribution.png'
        plot_distribution(df, distribution_plot_path, 'topic')
        count = 0
        word2vec_models_path = self.folder_path + self.algorithm + '/word2vec_models/'
        start_all_time = time.time()
        for category, df_category in df.groupby('topic'):
            start_time = time.time()
            count += 1
            model_name = word2vec_models_path + str(count) + ".model"
            word2vec_trainer(df=df_category, model_path=model_name)
            write_to_file("Time taken to train the word2vec model: " + str(time.time() - start_time) + '\n')
        write_to_file("Time taken to train all the word2vec models: " + str(time.time() - start_all_time) + '\n')
        write_to_file(80 * "#")

    def save_topic_model(self, model):
        model_path = self.folder_path + self.algorithm + '/model/' + self.algorithm + '.model'
        model.save(model_path)
        print("save_topic_model")

    def save_dict_and_tfidf(self):
        self.__save_dictionary()
        self.__save_tfidf_model()
        print("save_dict_and_tfidf")

    def search_num_of_topics(self):
        coherence_scores = []
        for i in [20, 30]:
        # for i in range(self.max_num_topics):
            print(i)
            model = self.get_model(i + 1)
            cm = CoherenceModel(model=model, texts=self.dataset,
                                corpus=self.corpus_tfidf, coherence='c_v')
            coherence_scores.append(cm.get_coherence())
        best_num_topics = coherence_scores.index(max(coherence_scores)) + 1
        print("best_num_topics: " + str(best_num_topics))
        best_model = self.get_model(best_num_topics)
        print("best_model retrieved")
        write_to_file(str(best_model.print_topics()))
        pprint(best_model.print_topics())
        self.save_topic_model(best_model)
        self.__plot_coherence_scores(coherence_scores)
        print("search_num_of_topics")
        return best_model

    def divide_into_clusters(self, best_model):
        topic_clusters = self.__extract_dominant_topics(best_model)
        extended_df = pd.DataFrame(list(zip(self.dataset, topic_clusters)), columns=['description', 'topic'])
        self.__extract_word2vec_models(extended_df)
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
        timing_log="training time of LSA model with "+str(num_topics)+" number of topics: "+str(int((time.time()-start_time)/60))+' minutes\n'
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
        timing_log="training time of LDA model with "+str(num_topics)+" number of topics: "+str(int((time.time()-start_time)/60))+' minutes\n'
        print(timing_log)
        write_to_file(timing_log)
        return lda_model


class HDP(TopicModel):

    def __init__(self, dataset, folder_path, algorithm):
        super().__init__(dataset, folder_path, algorithm)

    def get_model(self):
        start_time = time.time()
        hdp_model = gensim.models.hdpmodel.HdpModel(corpus=self.corpus_tfidf, id2word=self.dictionary)
        write_to_file('\n\n'+str(hdp_model.print_topics(num_words=10))+'\n\n')
        pprint(hdp_model.print_topics(num_words=10))
        print("training time of HDP model: " +str(int((time.time()-start_time)/60))+' minutes\n')
        write_to_file("Time taken to train the hdp model: "+str(int((time.time()-start_time)/60))+' minutes\n')
        self.save_topic_model(hdp_model)
        return hdp_model


def save_coherence_plot(max_num_topics, coherence_scores, figure_path):
    x = [i + 1 for i in [20, 30]]
    # x = [i + 1 for i in range(max_num_topics)]
    plt.figure(figsize=(10, 5))
    plt.plot(x, coherence_scores)
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence score')
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def pickle_save(my_model, file_name):
    pickle.dump(my_model, open(file_name, 'wb'))


def write_to_file(message):
    f = open('./output/topic_modeling/timings.txt', 'a')
    f.write(message)


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
    lsa_obj.save_dict_and_tfidf()
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
