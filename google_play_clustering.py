from w2v_trainer import *
from fast_text_trainer import *
try:
    from glove_trainer import *
except Exception as e:
    print(e)


def gp_cluster(df, model_path, word_embedding):
    plot_distribution(df, model_path + 'distribution.png', 'category')
    for category, df_category in df.groupby('category'):
        if word_embedding == "word2vec":
            model_name = model_path + "word2vec_models/"+"model_" + str(category)
            if not os.path.exists(model_path + "word2vec_models/"):
                os.makedirs(model_path + "word2vec_models/")
            word2vec_trainer(df_category, model_name)
        elif word_embedding == "fast_text":
            model_name = model_path + "fast_text_models/"+"model_"+str(category)
            if not os.path.exists(model_path + "fast_text_models/"):
                os.makedirs(model_path + "fast_text_models/")
            fast_text_trainer(df_category, model_name)
        elif word_embedding == "glove":
            try:
                glove_models_path = model_path + "glove_models/"
                if not os.path.exists(glove_models_path):
                    os.makedirs(glove_models_path)
                glove_trainer(df_category, glove_models_path, category)
            except Exception as e:
                print(e)

        else:
            print(word_embedding + " is not supported!")
            return


def main():
    conf = toml.load('config.toml')
    df = pd.read_csv(conf["preprocessed_data_path"])
    parser = argparse.ArgumentParser(description='Topic modeling software')
    parser.add_argument('--word_embedding', dest='word_embedding', type=str, help='word embedding algorithm')
    args = parser.parse_args()
    if args.word_embedding is None:
        args.word_embedding = "word2vec"
    gp_cluster(df, conf["google_play_model_path"], args.word_embedding)


if __name__ == "__main__":
    main()