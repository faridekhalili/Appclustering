# Setup
---
##### Requirements:

- virtualenv
- python3.8
- python3.8-dev
- python2.7
- python2.7-dev
- g++
- python-tk
- 32 GB RAM 

> **Note:** The 32GB RAM requirement is needed for the fast text word embedding models to be trained. The rest of the models can be trained with less.

The requirements can be installed with the following command.

```sh
sudo apt install virtualenv python3.8 python3.8-dev python2.7 python2.7-dev g++ python-tk
```
> **Note:** We need an environment **venv** created with python3.8 and requirements specified in the requirements.txt. We also need an environment **venv_glove** created with python2.7 and requirements specified in the requirements_glove.txt, which is necessary to train and use glove models. 

To set up a virtual environment, please follow the following steps:
1. create a virtual enviroment:
```sh
virtualenv -p /usr/bin/python3.8 venv
```
or
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
# Directory
---
First, you need to connect to the docker container with the following command:
```sh
docker exec -it semantic_matching /bin/bash
```
The following directory contains the data and models we used, and the code you can use to do additional experiments.
- root/ 
    1. Data/
    2.  Appclustering/

#### 1. Data:
---
The data and models used in the paper are stored in */root/Data* . In this directory, you can find the followings:

- **preprocessed.csv**: The GOOGLE-PLAY dataset after the canonical pre-processing steps.
- **labeled.csv**: The GOOGLE-PLAY dataset labeled with the most dominant topic of each application description document, using the LDA model introduced in the paper.
- *models/topic_modeling*: 
    - **lda.model**: The LDA model introduced in the paper.
    - **dictionary**:A mapping between words and their integer ids, based on the GOOGLE-PLAY dataset. It is required in training a topic model, estimating it's coherence score, and extracting the topic distribution of a document.
    - **tfidf_model**: Required in  extracting the topic distribution of a new document.
    - **tfidf_corpus**: The vectors of the TF-IDF values of the documents in GOOGLE-PLAY. It is required in training a topic model, estimating it's coherence score, and labeling the application description documents in GOOGLE-PLAY  dataset.
- *models/word_embedding_models*: The word embedding models trained based on the training sets introduced in the paper (first 5 rows of TABLE I). Under each of the following folders, there are three subfolders: *fast_text_models*, *glove_models*, *word2vec_models*.
    - *BLOGS*
    - *CATEGORIES*: Only the word embedding models trained on the categories of our subject applications are stored here.
    - *GOOGLE-PLAY*
    - *MANUALS*
    - *TOPICS*: Only the word embedding models trained on the topics of our subject applications are stored here.
    

> **Note:** Word Mover's distance (WM) uses word2vec vector embedding of words.

#### 2. Appclustering:
---
In order to reproduce the results, you need to follow the following steps:
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
        - *labeled.csv*: preprocessed.csv labeled with the dominant topic in the topic distribution of each of the application description documents.
3. Run word embedding trainer scripts (*glove_trainer.py*, *w2v_trainer.py*, *fast_text_trainer.py*)
    - Inputs:
        - The topic model with the highest coherence score.
        - *dictionary*
        - *tfidf_model*
        - *tfidf_corpus*
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

#### How to do experiments
---
In order to do customized experiments, please follow the following instructions:
> **Note:** First of all, you'll need to clear the data and models resultant from the previous experiments. You can do so with *clear* script:

```sh
chmod +x clear.sh
./clear.sh
```
> **Note:** In order to do experiments with different train sets, you'll need to provide a csv file in */root/Appclustering/output/*, and modify **preprocessed_data_path** specified in **config.toml**.
> **Note:** */root/Appclustering/output/sample.csv* is a small random sample of GOOGLE-PLAY.

##### Training a topic model

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
##### Retrieving the best topic model

First activate **venv**, then run the following command:
```sh
python best_topic_model_finder.py --algorithm "topic modeling algorithm"
```
In the command above:
1. *algorithm* should be "lda", "lsa", or "hdp". The default value is "lda".

This script will choose the best topic model automatically according to the coherence scores of the topic models trained in the previous step, and store it at the directory specified in **best_topic_model_path** of **config.toml**.

##### Training word embedding models

Word embedding models can be trained on subsets of the training datase, clustered by the best topic models trained in each of the topic modeling algorithms.

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

Three different folders with the name of the word embedding algorithms are created under the directory **best_topic_model_path** specified in **config.toml**.
##### Retrieving the best word embedding model

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

To load and use these models:
# UNFINISHED
#### CATEGORIES
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