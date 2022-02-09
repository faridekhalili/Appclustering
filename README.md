# Quick Start
---
We have set up a docker container that contains the results presented in the paper, and has the environment setup necessary to reproduce those results. 
First you need to import the docker image and run the container:
```sh
docker import sm.tar sm
docker run --name=semantic_matching sm sleep infinity
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
- *models/word_embedding_models*: This directory contains the word embedding models trained based on the training sets introduced in the paper (first 5 rows of TABLE I). Under each of the following folders, there are three subfolders: *fast_text_models*, *glove_models*, *word2vec_models* each of which contains the representative word embedding models.
    - *BLOGS*
    - *CATEGORIES*: Only the word embedding models trained on the categories of our subject applications are stored here.
    - *GOOGLE-PLAY*
    - *MANUALS*
    - *TOPICS*: Only the word embedding models trained on the topics of our subject applications are stored here.
    

> **Note:** Word Mover's distance (WM) uses word2vec vector embedding of words.

#### 2. Appclustering:
---

You can follow the following steps to reproduce the results:
> **Note:** The default values are set to produce the data and models presented in the paper. To produce these results, you can follow the following steps without worrying about the details. More detailed description is available in the next section. 

1. Activate **venv**:
 ```sh
source venv/bin/activate
```
2.
```sh
python topic_modeling.py
```
3.
```sh
python best_topic_model_finder.py
```
4.
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
6.
```sh
python glove_trainer.py
```
After runing the above commands the topic model should be stored in *./output/best_topic_model/lda/model/lda.model*. The word embedding models trained on subsets divided by this topic model will be stored in *./output/best_topic_model/lda/fast_text_models*, *./output/best_topic_model/lda/glove_models*, and *./output/best_topic_model/lda/word2vec_models*. 


# Setup
---
In order to do the necessary setup in a new environemt, follow the following instructions.
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

#### How to run new experiments
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
# UNFINISHED
# CATEGORIES
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