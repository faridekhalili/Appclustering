import pandas as pd
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LsiModel
from gensim.models import HdpModel
import toml

conf = toml.load('config.toml')
topic_modeling_path = conf['topic_modeling_path']
test_models_result_path = conf['test_models_result_path']

df = pd.read_csv(conf["preprocessed_data_path"])
df_length = len(df)
print(str(df_length) + " application descriptions were gathered in total by the crawler.\n\n",
      file=open(test_models_result_path, "a"))

lda_model_path = topic_modeling_path + 'lda/model/lda.model'
lda_model = LdaMulticore.load(lda_model_path)
lda_topics = lda_model.print_topics()
print(100 * "#", file=open(test_models_result_path, "a"))
print("examples of topics retrieved using LDA: \n", file=open(test_models_result_path, "a"))
for i in range(3):
    print(str(lda_topics[i]) + "\n", file=open(test_models_result_path, "a"))

lsa_model_path = topic_modeling_path + 'lsa/model/lsa.model'
lsa_model = LsiModel.load(lsa_model_path)
lsa_topics = lsa_model.print_topics()
print(100 * "#", file=open(test_models_result_path, "a"))
print("examples of topics retrieved using LSA: \n", file=open(test_models_result_path, "a"))
for i in range(3):
    print(str(lsa_topics[i]) + "\n", file=open(test_models_result_path, "a"))

hdp_model_path = topic_modeling_path + 'hdp/model/hdp.model'
topic_weight_df_path = topic_modeling_path + 'hdp/topic_wight_df.csv'
topic_wight_df = pd.read_csv(topic_weight_df_path)
hdp_model = HdpModel.load(hdp_model_path)
shown_topics = hdp_model.print_topics(num_topics=150, num_words=10)
print(100 * "#", file=open(test_models_result_path, "a"))
print("examples of topics retrieved using HDP: \n", file=open(test_models_result_path, "a"))
for i in range(3):
    print(str(shown_topics[int(topic_wight_df.iloc[i]["topic_id"])]) + "\n", file=open(test_models_result_path, "a"))
print(100 * "#", file=open(test_models_result_path, "a"))
