 MLOps-Project-Detox
==============================
Rune s202192 \
Jesper s133696 \
Daniel s103121 \
Gustav s215917

### Overall goal of the project
In this project we wish to classify the severity of toxic behaviour, in user generated Wikipedia comments, which have been flagged as having content containing toxic behaviour.

### Framework
As we are working with text data in the form of user generated comments, we have choosen to use the [Transformers](https://github.com/huggingface/transformers) framework for our project. This is in large part to due to the state of the art pre-trained languague models contained in the framework.

### How we intend to include the framework in our project
We will use the transformers framework by utilizing the available pre-trained language models. Using the framework we will encode the data into a representation that can be passed to a model from the Transformers framework.

### Data
We will be using a dataset from the Kaggle competetion [Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data). The dataset consists of user generated comments from Wikipedia. 

The train dataset contains a comment and a corresponding label (based on human rater annotations), which is in a binary format.

The Test dataset is split into a comments and labels file.

The toxic classification groups are as follows:

* `toxic`
* `severe_toxic`
* `obscene`
* `threat`
* `insult`
* `identity_hate`


### Deep learning models
We will utilize the Transformer framework's pre-trained models, and use these as a starting point. Ultimately we will see if we can fine tune the model to our specific dataset, and outperform the initial parameters of the model.

Specifically we will use the pre-trained `BERT` model. Hopefully we will have time too implement and deploy other models to our ML Pipeline, to see how well they perform in comparison to each other. Our current idea is to compare the smaller `DistilBERT` model. The goal would be to profile the models, and implement a method for comparing both the accuracy and computational performance of the model. The importance of accuracy is somewhat self explanatory, and the importance of the computational performance should be seen in light if the project scope, where we wish to implement an ML Pipeline that allows for re-training, evaluation, and re-deployment. We might be faced with an accuracy vs. computational performance tradeoff scenario, and as such it will be interresting to see if a smaller model (`DistilBERT`) can achieve an accuracy comparable to a bigger model (`BERT`), and at the same time hopefully have better computational perfomance.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── Docker
        └── pipeline.dockerfile
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
