import toml
import time
from ast import literal_eval
from glove import Corpus, Glove
from utils import *


def glove_trainer(df, model_path):
    list_of_tokens = list(df["description"])
    print(list_of_tokens[0])
    if isinstance(list_of_tokens[0], str):
        list_of_tokens = [literal_eval(x) for x in list_of_tokens]
    print(list_of_tokens[0])
    start_time = time.time()
    corpus = Corpus()
    corpus.fit(list_of_tokens, window=5)
    glove = Glove(no_components=60, learning_rate=0.01)

    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')
    print("Time taken to train the glove model: " + str(int((time.time() - start_time) / 60)) + ' minutes\n')
    glove.save(model_path)


def extract_glove_models(folder_path):
    extended_df = pd.read_csv(folder_path + 'labeled.csv')
    glove_models_path = folder_path + 'glove_models/'
    start_all_time = time.time()
    for category, df_category in extended_df.groupby('topic'):
        start_time = time.time()
        model_name = glove_models_path + "model_" + str(category)
        glove_trainer(df=df_category, model_path=model_name)
        write_to_file(
            "Time taken to train the " + str(category) + "th glove model: " +
            str(int((time.time() - start_time) / 60)) + ' minutes\n')
    write_to_file("Time taken to train all the glove models: " + str(
        int((time.time() - start_all_time) / 60)) + ' minutes\n\n')
    write_to_file(80 * "#" + '\n\n')
    print("extract_glove_models")


def main():
    conf = toml.load('config.toml')
    best_topic_model_path = conf['best_topic_model_path']
    parser = argparse.ArgumentParser(description='Topic modeling software')
    parser.add_argument("--modelNumbers", nargs="*")
    args = parser.parse_args()
    if args.modelNumbers is None:
        extract_glove_models(best_topic_model_path)
    else:
        extended_df = pd.read_csv(best_topic_model_path + '/labeled.csv')
        for model_number in args.modelNumbers:
            if int(model_number) > extended_df["topic"].max():
                print("Sorry there is no category " + model_number)
                continue
            start_time = time.time()
            model_name = best_topic_model_path + "/glove_models/model_" + model_number
            df_category = extended_df.loc[extended_df['topic'] == int(model_number)]
            glove_trainer(df=df_category, model_path=model_name)
            timing_log = "Time taken to train the " + model_number + ": " + \
                         str(int((time.time() - start_time) / 60)) + ' minutes\n'
            write_to_file(timing_log)


if __name__ == "__main__":
    main()
