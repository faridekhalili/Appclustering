# The Ineffectiveness of Domain Specific Word Embedding Models for GUI Test Reuse
F. Khalili, A. Mohebbi, V. Terragni, M. Pezze, L. Mariani, A. Heydarnoori

# Introduction
`Quick start` section provides the necessary steps to replicate the experiments of the paper. The section is structured as follow:
- `Data` explains locations of input data set and created LDA & Word Embedding models .
- `AppClustering` is dedicated to creating an LDA model and word embedding models. 
- `Semantic Matching` provides steps to use created word embedding models for semantic matching. Since the word embedding models already exist you can skip the `AppClustering` part to only replicate semantic matching experiments and get the results presented in the paper.

`Setup for new environment` section provides information to do the experiments in an environment other than released docker or change the current configurations. It also contains information to understand the workflow of the experiments in more detail. 

# Quick Start 
---
We have set up a docker container that contains the results presented in the paper, and has the environment setup necessary to reproduce those results. 
First you need to install [docker](https://docs.docker.com/get-docker/) if you don't have it. Then import the docker image and run the container:

```sh
docker import semantic_matching.tar semantic_matching
docker run --name=semantic_matching semantic_matching sleep infinity
```
Then you need to run the commands in the docker container, so you'll need to go inside the container:
```sh
docker exec -it semantic_matching /bin/bash
```
### Directory
---
- root/ 
    1. Data/
    2.  Appclustering/
    3.  semantic_matching/

## 1. Data:
---
The data and models used in the paper are stored in */root/Data* . In this directory, you can find the followings:

- **preprocessed.csv**: The GOOGLE-PLAY dataset after the canonical pre-processing steps.
- **labeled.csv**: The GOOGLE-PLAY dataset labeled with the most dominant topic of each application description document, using the LDA model introduced in the paper.
- *models/topic_modeling*: 
    - **lda.model**: The LDA model introduced in the paper.
    - **dictionary**:A mapping between words and their integer ids, based on the GOOGLE-PLAY dataset. It is required in training a topic model, estimating it's coherence score, and extracting the topic distribution of a document.
    - **tfidf_model**: Required in  extracting the topic distribution of a new document.
    - **tfidf_corpus**: The vectors of the TF-IDF values of the documents in GOOGLE-PLAY. It is required in training a topic model, estimating it's coherence score, and labeling the application description documents in GOOGLE-PLAY  dataset.
- *models/word_embedding_models*: This directory contains the word embedding models trained based on the training sets introduced in the paper (first 5 rows of TABLE I). Under each of the following folders, there are three subfolders: *fast_text_models*, *glove_models*, *word2vec_models* each of which contains the representative word embedding models.
    - *BLOGS*
    - *CATEGORIES*: Only the word embedding models trained on the categories of our subject applications are stored here.
    - *GOOGLE-PLAY*
    - *MANUALS*
    - *TOPICS*: Only the word embedding models trained on the topics of our subject applications are stored here.
    

> **Note:** Word Mover's distance (WM) uses word2vec vector embedding of words.

---
You can follow the following steps to reproduce the results:
> **Note:** The default values are set to produce the data and models presented in the paper. To produce these results, you can follow the following steps without worrying about the details. More detailed description is available in the next section. 

> **Note:** */root/Appclustering/output/sample.csv* is a small random sample of GOOGLE-PLAY.
In case that you don't have time, you can use the small sample to see how scripts can be run. To use the small data set modify **preprocessed_data_path** specified in **config.toml**. Obviously it will not produce the same results as the full data set.

## 2. AppClustering:
---


1. Activate **venv**:
 ```sh
source venv/bin/activate
```
2. Create an LDA topic modeling
```sh
python topic_modeling.py
```
3. Find the best model and labeling documents
```sh
python best_topic_model_finder.py
```
4. Train W2V and FAST embedding technique models for the topic model
```sh
python w2v_trainer.py
```
```sh
python fast_text_trainer.py
```

5. deactivate **venv** and activate **venv_glove**:
```sh
deactivate
```
```sh
source venv_glove/bin/activate
```
6. Train GloVe embedding technique models for the topic model
```sh
python glove_trainer.py
```

After runing the above commands the topic model are stored in `./output/best_topic_model/lda/model/lda.model`. The word embedding models trained on subsets divided by the topic model are stored in `./output/best_topic_model/lda/fast_text_models`, `./output/best_topic_model/lda/glove_models`, and `./output/best_topic_model/lda/word2vec_models`. 

#### CATEGORIES
---
1. Train W2V and FAST embedding technique models for the google categories

```sh
python google_play_clustering.py -- word_embedding "word2vec"
```
```sh
python google_play_clustering.py -- word_embedding "fast_text"
```

2. deactivate **venv** and activate **venv_glove**:
```sh
deactivate
```
```sh
source venv_glove/bin/activate
```
2. Train GloVe embedding technique models for the google categories
```sh
python google_play_clustering.py -- word_embedding "glove"
```


Runing this scripts results in word embedding models stored in **google_play_model_path** specified in **config.toml**.


## 3. semantic matching:
---

1. Activate virtual env
```shell
source venv/bin/activate
```
2. Run semantic matching
```shell
python run_all_combinations.py
```
3. check the results
    - MRR and top1 values in available the `final.csv`.
    - Results of the table in the paper are available in `table_mrr.csv` and `tabel_top1.csv`
> Semantic matching uses the models provided in the `Data` directory. If you like to use newly created models in the past steps. You have to move them to `Data` directory recpecting the existing structure. 
# Setup for new  environment
---
In order to do the necessary setup in a new environemt, follow the following instructions.
### Requirements:

- virtualenv
- python3.8
- python3.8-dev
- python2.7
- python2.7-dev
- python3.7
- python3.7-dev
- build-essential
- libssl-dev
- g++
- python-tk
- 32 GB RAM 

> **Note:** The 32GB RAM requirement is needed for the fast text word embedding models to be trained. The rest of the models can be trained with less.

The requirements can be installed with the following command.

```sh
sudo apt install virtualenv python3.8 python3.8-dev python2.7 python2.7-dev g++ python-tk python3.7 python3.7-dev build-essential libssl-dev
```
> **Note:** We need an environment **venv** created with python3.8 and requirements specified in the requirements.txt. We also need an environment **venv_glove** created with python2.7 and requirements specified in the requirements_glove.txt, which is necessary to train and use glove models. 

To set up a virtual environment, please follow the following steps:
1. Create the virtual enviroments:
```sh
virtualenv -p /usr/bin/python3.8 venv
```
```sh
virtualenv -p /usr/bin/python2.7 venv_glove
```
2. Activate the environment:
```sh
source venv/bin/activate
```
or
```sh
source venv_glove/bin/activate
```
To deactivate an environemt:
```sh
deactivate
```
In order to reproduce the represented models and results, you need to follow the following steps:
1. Run *topic_modeling.py*
    - Inputs:
        - *preprocessd.csv*
    - Outputs:
        - A csv file with c_v values of the trained topic models.
        - *dictionary*
        - *tfidf_model*
        - *tfidf_corpus*
2. Run *best_topic_model_finder.py*
    - Inputs:
        - The csv files with c_v values of the trained topic models.
        - *dictionary*
        - *tfidf_model*
        - *tfidf_corpus*
        - *preprocessed.csv*
    - Outputs:
        - The topic model with the highest coherence score.
        - *labeled.csv*: The preprocessed train set labeled with the dominant topic in the topic distribution of each of the application description documents.
3. Run word embedding trainer scripts (*glove_trainer.py*, *w2v_trainer.py*, *fast_text_trainer.py*)
    - Inputs:
        - *labeled.csv*
    - Outputs:
        - Word embedding models trained on each of the subsets divided by topics.
4. Semantic matching
    - Inputs:
        - Word embedding models.
        - events.
    - Outputs:
        - Metric.
5. Result aggregator
    - Inputs:
    - Outputs:

# How to run new experiments
---
In order to do customized experiments, please follow the following instructions:
> **Note:** First of all, you'll need to clear the data and models resultant from the previous experiments. You can do so with *clear* script:

```sh
chmod +x clear.sh
./clear.sh
```
> **Note:** In order to do experiments with different train sets, you'll need to provide a csv file in */root/Appclustering/output/*, and modify **preprocessed_data_path** specified in **config.toml**.


### Training topic models
---

First activate **venv**, then run the following command:
```sh
python topic_modeling.py --algorithm "topic modeling algorithm" --word_filter "vocabulary pruning strategy" --document_filter "document pruning strategy" --min "min number of topics" --max "max number of topics" --step "the step to change the number of topics"
```
In the command above:
1. *algorithm* should be "lda", "lsa", or "hdp". The default value is "lda".
2. *word_filter* should be "s1" or "s2". The default value is "s1".
3. *document_filter* should be "s3", "s4", or "s5". The default value is "s5".
4. *min*, *max*, and *step* determines number of topics in the models being trained. The default values are set so that only one topic model with 27 number of topics is trained.

> **Note:** The vocabulary and document pruning strategies have default values set in **utils. py**, **filter_words** and **filter_documents** functions. Feel free to do additional experiments with modifying the default values.

The output of this script is a csv file that contain coherence scores of the trained models, which will be stored in  **topic_modeling_path** specified in **config.toml**.
### Retrieving the best topic model:

First activate **venv**, then run the following command:
```sh
python best_topic_model_finder.py --algorithm "topic modeling algorithm"
```
In the command above:
1. *algorithm* should be "lda", "lsa", or "hdp". The default value is "lda".

This script will choose the best topic model automatically according to the coherence scores of the topic models trained in the previous step, and store it at the directory specified in **best_topic_model_path** of **config.toml**.

### Training word embedding models
---

Word embedding models can be trained on subsets of the training dataset, clustered by the best topic models trained in each of the topic modeling algorithms.

To train word embedding models trained on these subsets seperated by the topics:

1. If you want to train word2ec, or fast text models, activate **venv**, then run one of the following commands:
```sh
python fast_text_trainer.py --algorithm "topic modeling algorithm" --modelNumber "word embedding model number"
```
or
```sh
python w2v_trainer.py --algorithm "topic modeling algorithm" --modelNumber "word embedding model number"
```
2. If you want to train glove models, activate **venv_glove**, then run the following command:
```sh
python glove_trainer.py --algorithm "topic modeling algorithm" --modelNumber "word embedding model number"
```
In the above commands:
1. *algorithm* should be "lda", "lsa", or "hdp". The default value is "lda".
2. *modelNumber* is the number of the subset (topic) you want to train a model on. The default value is set so that a word embedding model is trained on each and every one of the subsets.

The word embedding models are stored under three different folders named with the word embedding algorithm under the directory **best_topic_model_path** specified in **config.toml**.

### Retrieving the best word embedding model

To retrieve the best word embedding model for a given source application, we need to:
1. Find the most fit model's address.
2. Load the model.

To retrieve the address, first activate **venv**, then run the following command:
```sh
python best_word_embedding_model_finder.py --algorithm "topic modeling algorithm" --word_embedding "word embedding algorithm" --app_name "The name of the source application"
```
In the command above:
1. *algorithm* should be "lda", "lsa", or "hdp". The default value is "lda".
2. *word_embedding* should be "word2vec", "fast_text", or "glove". The default value is "word2vec".
3. *app_name* should be one of the subject application names specidied in */roor/Appclustering/input/app_name_to_id.csv*.

This query returns the most fit word embedding model's path, and stores it in a txt file under the **query_result_path** directory specified in **config.toml**.

In order to load and use these word embedding models you need to have the following lines of code in your script:
- fast_text
```sh
from gensim.models import FastText
fast_text_20 = FastText.load([MODEL_PATH])
```
- word2vec
```sh
from gensim.models import Word2Vec
word2vec_20 = Word2Vec.load([MODEL_PATH])
```
- glove 
```sh
from gensim.models import KeyedVectors
glove_20 = KeyedVectors.load_word2vec_format([MODEL_PATH])
```
In the above commands the MODEL_PATH is the path stored in the txt file generated in step 1.

### CATEGORIES
---
In order to train the word embdding models on subsets of GOOGLE-PLAY training set, divided by the googleplay's category metadata:

- If you want to train "word2vec" or "fast text" models, activate **venv**.
- If you want to train "glove" models, activate **venv_glove**

Then run the following command:
```sh
python google_play_clustering.py -- word_embedding "word embedding algorithm"
```
In the command above:
1. *word_embedding* should be "word2vec", "fast_text", or "glove". The default value is "word2vec".

Runing this scripts results in word embedding models stored in **google_play_model_path** specified in **config.toml**.


### Semantic Matching
---
1. Create a virtual environment:
```sh
virtualenv -p /usr/bin/python3.7 venv
```


2. Activate the environment:
```sh
source venv/bin/activate
```

3. Install required packages

```shell
pip install -r requirements.txt
```

4. Modify `config.yml` following entry:
    - `model_dir` : path to the word embedding models with respect to `model_path` entry

5. In case of using a new LDA model, `embedding/app_to_cluster.csv` should be updated. You can update it manually or by help of `app_to_model_mapper.py`.

> If you like to use `app_to_model_mapper.py`, you have to provide address of the LDA model in `config.yml` in the `topic_model` entry
6. Remove the files inside `sim_scores`. Then run `clean.sh`.
7. Run semantic matching

```shell
python run_all_combinations.py
```

8. check the results
    - MRR and top1 values in available the `final.csv`.
    - Results of the table in the paper are available in `table_mrr.csv` and `tabel_top1.csv`

