import gensim
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
from Categorization.Word2Vec import *
from abc import ABC, abstractmethod


class TopicModel(ABC):
    def __init__(self, dataset, folder_path, algorithm: str):
        self.__max_num_topics = 100
        self.__dataset = dataset
        self.__folder_path = folder_path
        self.__algorithm = algorithm
        self.__dictionary = gensim.corpora.Dictionary(self.__dataset)
        bow_corpus = [self.__dictionary.doc2bow(doc) for doc in self.__dataset]
        tfidf = gensim.models.TfidfModel(bow_corpus)
        self.tfidf = tfidf
        self.__corpus_tfidf = tfidf[bow_corpus]

    def __save_dictionary(self):
        dictionary_path = self.__folder_path + "dataset.dict"
        self.__dictionary.save(dictionary_path)

    def __save_tfidf_corpus(self):
        tfidf_path = self.__folder_path + "dataset.tfidf_model"
        self.tfidf.save(tfidf_path)

    def __save_topic_model(self, model):
        model_path = self.__folder_path + self.__algorithm + '/model/' + self.__algorithm + '.model'
        model.save(model_path)

    def __plot_coherence_scores(self, coherence_scores):
        figure_path = self.__folder_path + self.__algorithm + '/' + self.__algorithm + '_coherence.png'
        save_coherence_plot(self.__max_num_topics, coherence_scores, figure_path)

    def __extract_dominant_topics(self, best_model):
        topic_clusters = []
        for i in range(len(self.__corpus_tfidf)):
            topic_distribution = dict(best_model[self.__corpus_tfidf[i]])
            dominant_topic = max(topic_distribution, key=topic_distribution.get)
            topic_clusters.append(dominant_topic)
        return topic_clusters

    def __extract_word2vec_models(self, df):
        distribution_plot_path = self.__folder_path + self.__algorithm + '/topic_distribution.png'
        plot_distribution(df, distribution_plot_path, 'topic')
        count = 0
        word2vec_models_path = self.__folder_path + self.__algorithm + '/word2vec_models/'
        for category, df_category in df.groupby('topic'):
            count += 1
            model_name = word2vec_models_path + str(count) + ".model"
            word2vec_trainer(df=df_category, size=60, model_path=model_name)

    @abstractmethod
    def __get_model(self, num_topics):
        pass

    def save_dict_and_tfidf(self):
        self.__save_dictionary()
        self.__save_tfidf_corpus()

    def search_num_of_topics(self):
        coherence_scores = []
        for i in range(self.__max_num_topics):
            model = self.__get_model(i + 1)
            cm = CoherenceModel(model=model, texts=self.__dataset,
                                corpus=self.__corpus_tfidf, coherence='c_v')
            coherence_scores.append(cm.get_coherence())
        best_num_topics = coherence_scores.index(max(coherence_scores)) + 1
        best_model = self.__get_model(best_num_topics)
        pprint(best_model.print_topics())
        self.__save_topic_model(best_model)
        self.__plot_coherence_scores(coherence_scores)
        return best_model

    def divide_into_clusters(self, best_model):
        topic_clusters = self.__extract_dominant_topics(best_model)
        extended_df = pd.DataFrame(list(zip(list(self.__dataset["description"]),
                                            topic_clusters)),
                                   columns=['description', 'topic'])
        self.extract_word2vec_models(extended_df)

    
class LSA(TopicModel):

    def __init__(self, dataset, folder_path, algorithm: str):
        super().__init__(dataset, folder_path, algorithm)

    def __get_model(self, num_topics):
        lsa_model = gensim.models.LsiModel(self.__corpus_tfidf,
                                           num_topics=num_topics,
                                           id2word=self.__dictionary)
        return lsa_model


class LDA(TopicModel):
    def __init__(self, dataset, folder_path, algorithm: str):
        super().__init__(dataset, folder_path, algorithm)

    def __get_model(self, num_topics):
        lda_model = gensim.models.LdaMulticore(self.__corpus_tfidf,
                                               num_topics=num_topics,
                                               id2word=self.__dictionary,
                                               passes=10, workers=10, iterations=100)
        return lda_model


def save_coherence_plot(max_num_topics, coherence_scores, figure_path):
    x = [i + 1 for i in range(max_num_topics)]
    plt.figure(figsize=(10, 5))
    plt.plot(x, coherence_scores)
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence score')
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.show()


def main():
    conf = toml.load('../config-temp.toml')
    topic_modeling_path = '../' + conf['topic_modeling_path']
    df = pd.read_csv('../' + conf["preprocessed_data_path"])
    texts = list(df["description"])

    lsa_obj = LSA(texts, topic_modeling_path, "lsa")
    best_lsa_model = lsa_obj.search_num_of_topics()
    lsa_obj.save_dict_and_tfidf()
    lsa_obj.divide_into_clusters(best_lsa_model)

    lsa_obj = LDA(texts, topic_modeling_path, "lda")
    best_lda_model = lsa_obj.search_num_of_topics()
    lsa_obj.divide_into_clusters(best_lda_model)


if __name__ == "__main__":
    main()
