# Setup
---
First, you need to connect to the docker container with the following command:
```sh
docker exec -it semantic_matching /bin/bash
```

### Directory
---
- home -> Appclustering/

The GOOGLE-PLAY database, the best topic model trained on it, and all the word embedding models trained on TOPICS dataset are stored in the container under the representative directories specified in **config.toml**. To do additional experiments, please follow the following instructions.
> **Note:** The word embdding models are stored in different folders labeled with the word embedding algorithm (glove, word2vec, fast text), under the **best_topic_model_path** directory specified in **config.toml**.

### Requirements:
---

- Python3.8
- python3.8-dev
- Python2.7
- python2.7-dev

These are already installed in the container.
> **Note:** We need an environment **venv** created with python3.8 and requirements specified in the requirements.txt. We also need an environment **venv_glove** created with python3.8 and requirements specified in the requirements_glove.txt.
#### Basic preproessing
---
The GOOGLE-PLAY database is stored in the **database_path** directory specified in **config.toml**. To apply the basic preprocessing on the database, first conect to **venv**:
```sh
source venv/bin/activate
```
Then run the following command:
```sh
python preprocessor.py
```
The result will be stored in **preprocessed_data_path** directory specified in **config.toml**.

#### Training a topic model
---
First connect to **venv**:
```sh
source venv/bin/activate
```
Then run the following command:
```sh
python topic_modeling.py --algorithm "topic modeling algorithm" --word_filter "vocabulary pruning strategy" --document_filter "document pruning strategy" --min "min number of topics" --max "max number of topics" --step "the step to change the number of topics"
```
In the command above:
1. *algorithm* should be "lda", "lsa", or "hdp". The default value is "lda".
2. *word_filter* should be "s1" or "s2". The default value is "s1".
3. *document_filter* should be "s3", "s4", or "s5". The default value is "s5".
4. *min*, *max*, and *step* determines number of topics in the models being trained. The default values are set so that only one topic model with 27 number of topics is trained.

> **Note:** The vocabulary and document pruning strategies have default values set in **utils. py**, **filter_words** and **filter_documents** functions. Feel free to do additional experiments with modifying the default values.

The trained topic models will be stored at the directory specified by **topic_modeling_path** in config.toml.
#### Retrieving the best topic model
---
First connect to **venv**:
```sh
source venv/bin/activate
```
Then run the following command:
```sh
python best_topic_model_finder.py --algorithm "topic modeling algorithm"
```
In the command above:
1. *algorithm* should be "lda", "lsa", or "hdp". The default value is "lda".

The best topic model will be stored at the directory specified in **best_topic_model_path** in **config.toml**.

#### Training word embedding models
---
Word embedding models can be trained on subsets of the GOOGLE-PLAY datase, clustered by the best topic model trained in each of the topic modeling algorithms.

To train word embedding models trained on the TOPICS dataset:

1. If you want to train word2ec, or fast text models, connect to **venv**:
 ```sh
source venv/bin/activate
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Then run one of the following commands:
```sh
python fast_text_trainer.py --algorithm "topic modeling algorithm" --modelNumber "word embedding model number"
```
or
```sh
python w2v_trainer.py --algorithm "topic modeling algorithm" --modelNumber "word embedding model number"
```
2. If you want to train glove models, connect to **venv_glove**:
 ```sh
source venv_glove/bin/activate
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Then run the following command:
```sh
python glove_trainer.py --algorithm "topic modeling algorithm" --modelNumber "word embedding model number"
```
In the above commands:
1. *algorithm* should be "lda", "lsa", or "hdp". The default value is "lda".
2. *modelNumber* is the number of the model you want to train. The default value is set so that all word embedding models be trained on subsets.

Three different folders with the name of the word embedding algorithms are created under the directory specified with **best_topic_model_path** in **config.toml**.
#### Retrieving the best word embedding model

To retrieve the best word embedding model for a given source application, first connect to **venv**:
```sh
source venv/bin/activate
```
Then run the following command:
```sh
python best_word_embedding_model_finder.py --algorithm "topic modeling algorithm" --word_embedding "word embedding algorithm" --app_name "The name of the source application"
```
In the command above:
1. *algorithm* should be "lda", "lsa", or "hdp". The default value is "lda".
2. *word_embedding* should be "word2vec", "fast_text", or "glove". The default value is "word2vc".
3. *app_name* should be on of the subject application names specidied in */Appclustering/input/app_name_t_id.csv*.

This query returns the most fit word embedding model's path, and stores it in a txt file under the **query_result_path** directory specified in **config.toml**.
