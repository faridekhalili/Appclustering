from word2vec import *


def gp_cluster(df, model_path):
    plot_distribution(df, model_path + '../distribution.png', 'category')
    count = 0
    for category, df_category in df.groupby('category'):
        count += 1
        model_name = model_path + str(count) + "th_model.model"
        word2vec_trainer(df_category, model_name)


def main():
    conf = toml.load('config.toml')
    df = pd.read_csv(conf["preprocessed_data_path"])
    gp_cluster(df, conf["google_play_model_path"])


if __name__ == "__main__":
    main()