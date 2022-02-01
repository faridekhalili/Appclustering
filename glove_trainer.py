import time
from ast import literal_eval
from glove import Corpus, Glove
from utils import *
import os


def write_word2vec_format(model, path):
    f = open(path, 'a')
    words = list(model.dictionary.keys())
    f.write(str(len(words)) + " ")
    f.write(str(len(model.word_vectors[model.dictionary[words[0]]])))
    f.write("\n")
    for i in range(len(words)):
        current_word = words[i]
        current_word_vector = model.word_vectors[model.dictionary[current_word]]
        f.write(current_word)
        for j in range(len(current_word_vector)):
            f.write(" " + str(current_word_vector[j]))
        f.write("\n")


def glove_trainer(df, model_path, model_number):
    model_name = model_path + "model_" + str(model_number)
    list_of_tokens = list(df["description"])
    # print(list_of_tokens[0])
    if isinstance(list_of_tokens[0], str):
        # list_of_tokens = [literal_eval(x) for x in list_of_tokens]
        tokenized_data = df[['description']].applymap(lambda s: word_tokenize(s))
        list_of_tokens = list(tokenized_data["description"])
    # print(list_of_tokens[0])
    start_time = time.time()
    corpus = Corpus()
    corpus.fit(list_of_tokens, window=5)
    glove = Glove(no_components=60, learning_rate=0.01)

    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save(model_name)
    if not os.path.exists(model_path+"/word2vec_format/"):
        os.makedirs(model_path+"/word2vec_format/")
    write_word2vec_format(glove, model_path+"/word2vec_format/" + str(model_number) + ".txt")
    print("Time taken to train the glove model: " + str(int((time.time() - start_time) / 60)) + ' minutes\n')


def extract_glove_models(folder_path, algorithm):
    extended_df = pd.read_csv(folder_path + algorithm+ '/labeled.csv')
    glove_models_path = folder_path + algorithm+ '/glove_models/'
    if not os.path.exists(glove_models_path):
        os.makedirs(glove_models_path)
    start_all_time = time.time()
    for category, df_category in extended_df.groupby('topic'):
        start_time = time.time()
        glove_trainer(df=df_category, model_path=glove_models_path,
                      model_number=category)
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
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='topic modeling algorithm')
    args = parser.parse_args()
    if args.algorithm is None:
        args.algorithm = "lda"
    if args.modelNumbers is None:
        extract_glove_models(best_topic_model_path, args.algorithm)
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
