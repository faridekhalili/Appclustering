from Categorization.Word2Vec import *


def gp_cluster(df, model_path):
    count = 0
    for category, df_category in df.groupby('category'):
        count += 1
        model_name = model_path + str(count) + "th_model.model"
        model = word2vec_trainer(df_category)
        model.save(model_name)


def main():
    conf = toml.load('../config-temp.toml')
    df = pd.read_csv('../'+conf["preprocessed_data_path"])
    gp_cluster(df, '../'+conf["google_play_model_path"])


if __name__ == "__main__":
    main()