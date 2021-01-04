import toml
import time
from gensim.models import Word2Vec
from ast import literal_eval
from gensim.models.callbacks import CallbackAny2Vec
from utils import *


class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'  Loss: {loss}')
        self.epoch += 1

    def per_epoch_loss(self):
        per_epoch_loss = []
        for i in range(len(self.losses)):
            if i == 0:
                per_epoch_loss.append(self.losses[i])
            else:
                per_epoch_loss.append(self.losses[i] - self.losses[i - 1])
        return per_epoch_loss


def word2vec_trainer(df, model_path):
    list_of_tokens = list(df["description"])
    if isinstance(list_of_tokens[0], str):
        list_of_tokens = [literal_eval(x) for x in list_of_tokens]
    start_time = time.time()
    loss_logger = LossLogger()
    model = Word2Vec(sentences=list_of_tokens,
                     sg=1,
                     size=300,
                     window=10,
                     min_count=1,
                     callbacks=[loss_logger],
                     compute_loss=True,
                     workers=3,
                     iter=20)
    print("Time taken to train the word2vec model: " + str(int((time.time() - start_time) / 60)) + ' minutes\n')
    model.save(model_path)


def extract_word2vec_models(folder_path, algorithm):
    extended_df = pd.read_csv(folder_path + algorithm + '/labeled.csv')
    word2vec_models_path = folder_path + algorithm + '/word2vec_models/'
    start_all_time = time.time()
    for category, df_category in extended_df.groupby('topic'):
        start_time = time.time()
        model_name = word2vec_models_path + "w2v_model_"+str(category)
        word2vec_trainer(df=df_category, model_path=model_name)
        write_to_file(
            "Time taken to train the " + str(category) + "th word2vec model resulting from " + str(
                algorithm) + ": " + str(
                int((time.time() - start_time) / 60)) + ' minutes\n')
    write_to_file("Time taken to train all the word2vec models: " + str(
        int((time.time() - start_all_time) / 60)) + ' minutes\n\n')
    write_to_file(80 * "#" + '\n\n')
    print("extract_word2vec_models")


def main():
    conf = toml.load('config.toml')
    topic_modeling_path = conf['topic_modeling_path']
    parser = argparse.ArgumentParser(description='Topic modeling software')
    parser.add_argument("--modelNumbers", nargs="*")
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='topic modeling algorithm')
    args = parser.parse_args()
    if args.algorithm is None:
        print('set algorithm first')
        return
    if args.modelNumbers is None:
        extract_word2vec_models(topic_modeling_path, args.algorithm)
    else:
        model_path = topic_modeling_path + args.algorithm
        extended_df = pd.read_csv(model_path + '/labeled.csv')
        for model_number in args.modelNumbers:
            if int(model_number) > extended_df["topic"].max():
                print("Sorry there is no category " + model_number + " created from the " + args.algorithm)
                continue
            start_time = time.time()
            model_name = model_path + "/word2vec_models/" + model_number + ".model"
            df_category = extended_df.loc[extended_df['topic'] == int(model_number)]
            word2vec_trainer(df=df_category, model_path=model_name)
            timing_log = "Time taken to train the " + model_number + \
                         "th word2vec model resulting from " + str(args.algorithm) + ": " + \
                         str(int((time.time() - start_time) / 60)) + ' minutes\n'
            write_to_file(timing_log)


if __name__ == "__main__":
    main()
