import toml
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


def plot_distribution(df, plot_path, col):
    plt.figure(figsize=(15, 5))
    pd.value_counts(df[col]).plot.bar(title="category distribution in the dataset")
    plt.xlabel("category")
    plt.ylabel("Number of applications in the dataset")
    plt.savefig(plot_path)


def word2vec_trainer(df, size, model_path):
    start_time = time.time()
    model = Word2Vec(list(df["description"]),
                     min_count=1, size=size, workers=3, window=3, sg=1)
    print("Time taken to train the word2vec model: " + str(time.time() - start_time))
    model.save(model_path)


def write_w2vec_vectors(word2vec_filename, df, w2v_model, w2v_vector_size):
    with open(word2vec_filename, 'w+') as word2vec_file:
        for index, row in df.iterrows():
            model_vector = (np.mean([w2v_model[token] for token in row['description']], axis=0)).tolist()
            if index == 0:
                header = ",".join(str(ele) for ele in range(w2v_vector_size))
                word2vec_file.write(header)
                word2vec_file.write("\n")
            # Check if the line exists else it is vector of zeros
            if type(model_vector) is list:
                line1 = ",".join([str(vector_element) for vector_element in model_vector])
            else:
                line1 = ",".join([str(0) for i in range(w2v_vector_size)])
            word2vec_file.write(line1)
            word2vec_file.write('\n')


def main():
    conf = toml.load('../config-temp.toml')
    df = pd.read_csv('../' + conf["preprocessed_data_path"])
    model_path = '../' + conf['model_path']
    word2vec_trainer(df, 60, model_path)


if __name__ == "__main__":
    main()
