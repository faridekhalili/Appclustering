import toml
import time
from gensim.models.fasttext import FastText
from ast import literal_eval
from utils import *


def fast_text_trainer(df, model_path):
    list_of_tokens = list(df["description"])
    if isinstance(list_of_tokens[0], str):
        list_of_tokens = [literal_eval(x) for x in list_of_tokens]
    start_time = time.time()
    model = FastText(sentences=list_of_tokens,
                     sg=1,
                     size=60,
                     window=5,
                     min_count=1,
                     compute_loss=True,
                     workers=3,
                     iter=30)
    print("Time taken to train the fast_text model: " + str(int((time.time() - start_time) / 60)) + ' minutes\n')
    model.save(model_path)


def extract_fast_text_models(folder_path):
    extended_df = pd.read_csv(folder_path + 'labeled.csv')
    fast_text_models_path = folder_path + 'fast_text_models/'
    start_all_time = time.time()
    for category, df_category in extended_df.groupby('topic'):
        start_time = time.time()
        model_name = fast_text_models_path + "model_" + str(category)
        fast_text_trainer(df=df_category, model_path=model_name)
        write_to_file(
            "Time taken to train the " + str(category) + "th fast_text model: " +
            str(int((time.time() - start_time) / 60)) + ' minutes\n')
    write_to_file("Time taken to train all the fast_text models: " + str(
        int((time.time() - start_all_time) / 60)) + ' minutes\n\n')
    write_to_file(80 * "#" + '\n\n')
    print("extract_fast_text_models")


def main():
    conf = toml.load('config.toml')
    best_topic_model_path = conf['best_topic_model_path']
    parser = argparse.ArgumentParser(description='Topic modeling software')
    parser.add_argument("--modelNumbers", nargs="*")
    args = parser.parse_args()
    if args.modelNumbers is None:
        extract_fast_text_models(best_topic_model_path)
    else:
        extended_df = pd.read_csv(best_topic_model_path + '/labeled.csv')
        for model_number in args.modelNumbers:
            if int(model_number) > extended_df["topic"].max():
                print("Sorry there is no category " + model_number)
                continue
            start_time = time.time()
            model_name = best_topic_model_path + "/fast_text_models/model_" + model_number
            df_category = extended_df.loc[extended_df['topic'] == int(model_number)]
            fast_text_trainer(df=df_category, model_path=model_name)
            timing_log = "Time taken to train the " + model_number + ": " + \
                         str(int((time.time() - start_time) / 60)) + ' minutes\n'
            write_to_file(timing_log)


if __name__ == "__main__":
    main()
