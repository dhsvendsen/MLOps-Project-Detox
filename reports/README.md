---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [ ] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [x] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Make sure all group members have a understanding about all parts of the project
* [x] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- question 1 fill here ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s215917

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

Since our project is a NLP classification task, we utilized the Transformers framework from Huggingface. More specifically, we used the BertModel with pretrained weights. 

First we preprocess the data by stemming, lowering and removing stop words using functionality from NLTK. Then we tokenize the comments using the functionality from the BertTokenizer. The new dataset is split into test, train and validation sets and saved as a tensor. 

We then implemented a PyTorch NN model whose first layers are the BertModel with  frozen weights, followed by a trainable dense classification layer. As such, the Transformers framework helped us complete the project by effectively functioning as an embedding tool, whose outputs could then be classified with a simple linear layer. 

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

The first step would be to clone our repository from github:

```bash
git clone https://github.com/dhsvendsen/MLOps-Project-Detox.git
```

This repository contains a set of docker images and a requirements file (`requirements.txt`). To get an exact copy of our environment, one could simply create a virtual environment and pip install the contents of `requirements.txt`:
```bash
pip install -r requirements.txt
```
However, to get a higher degree of reproducability, it is recommened to instead utilize the docker files, for building docker images. The dockerfiles are stored in github, where we have `TrainerLocal.dockerfile` for the training, and `Simple_app.dockerfile` for the FastApi deployment. You can then build the image locally, eg.:
```bash
docker build -f Simple_app.dockerfile . -t train:latest
```
then tag and push the image to your GCP conntainer registry:
```bash
docker tag train gcr.io/<project_id>/train
docker push gcr.io/<project_id>/train
```
and you would now be able to run the image in GCP

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

Hundreds of different cookiecutter templates exist, due to the nature of the project we decided to initialise our project using the cookiecutter "data-science-template" which we cloned from https://github.com/drivendata/cookiecutter-data-science. From this template we have removed the References folder as it was unused in our project. A .dvc folder has been initialised to keep the dvc config files that enable users to pull the data from our remote storage, and we have added docker and cloudbuild files to our root directory. When running the code locally the user needs to call dvc data pull, which will create a data folder locally, as well as the dvc model pull which will clone the latest model. In the root of the folder added dockerfiles, makefiles and a cloudbuild file. Using the cookiecutter data science template makes it easier for others to understand the logic of our project and quickly familiarise them with the codebase.


### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

In terms of quality we made sure to run “black .” pretty regularly and talked continuously about how to structure our code. In larger companies when people don’t sit close together as we did, it makes sense that code must be very readable so that people don’t waste time trying to decipher each other. A good thing we could have done is to write more function documentation. Maybe also to have used precommit for even cleaner code.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we have implemented 6 tests. Primarily we are testing whether the data is balanced but also its shape. While it helped using the “pos_weight” in “bcelosswithlogits” to counteract class-imbalance, the best solution was to balance the data directly. We tested the model as well in terms of being able to predict, and outputting tensors of the right shape. We also added a “raise ValueError” in our models forward function.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

We calculated coverage pretty late and ran into “No source for code: '/xxxxx/_remote_module_non_scriptable.py'.” and didn’t manage to debug. Comparing to when we did the exercises (98-100%) we probably also hit a decent coverage on the source code we tested which is the model and data but not the training/predicting scripts. Clearly, it is possible to write a test that executes the entire model class and therefore gets high coverage. While this shows that the code at least runs without throwing errors, it does not mean that all is well necessarily: e.g. we had a version “make_dataset” that was making data of the right shape, but we had interpreted the data description on kaggle wrong (what the flag -1 meant), leading to bad results. Similarly a model can run and output results even if someone changes loss = loss*0, etc. Tests don’t guard against bad math or stupidity, but maybe against silly mistakes.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

In our project we utilized branches in our workflow in order to be able to work on the project simultaneously. Each member had one or more branch(es) which the person worked on. 

To give an example, one branch (transformer_development) was dedicated to creating a make_dataset script, which transformed the raw data into tensors that the model could load during the training time. When this script was finished, we pulled from main into the transformer_develop and resolved any conflicts. We would then do one of two things: 

Either we used 'git checkout main' to merge the branches locally and then subsequently push to the remote main branch. At other times, we comitted the local changes on the developer branch and created a pull request to main. By accepting this pull request, the two branches were merged and the make_dataset script added to the main branch. Similarly, for the other parts of the project. The former approach is likely the more appropriate for a group project, whereas the latter is representative of how one would contribute to a larger project that one does not have full permission over.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC for managing both our data (raw and processed) as well as our model checkpoint. The actual data was stored in a Google Bucket with open read-access. This means that it is fairly easy for anyone with access to our git repo, to pull the repo, call DVC pull and then receive the data and model files needed to build the docker images for both training and inference. However, the process of updating our DVC model files got complicated when running our training on GCE, since we did not have git authentication on the VM. This meant that we could not correctly add and update the .dvc files in our repo from our VM, and instead we saved the model from our VM to google bucket by using the Google.storage library. As a consequence, the model file is initially not DVC tracked, and can not be automatically pulled by others by calling dvc pull. Hence, when updating the model, one afterwards locally has to download the new model from the bucket on a laptop with our git repo initialised, and then manually add the new model with dvc add, dvc push and git push the changed .dvc files.


### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

We use github actions to check pep8 compliance and run tests. We have two workflow files for running isort and flake8 respectively and one to run our tests using pytest. The workflow responsible for testing runs them on both windows, linux and macos but only with python 3.9. It seemed that using caching didn’t speed up the process so we did not use it.
We only implemented the pytest workflow quite late, but the linting workflow helped us see when one of us had forgotten to run flake8 to check for pep8 compliance before pushing. This made it a bit more obvious when someone had pushed and whether they had been a bit lazy with the code. 
We didn’t use github actions to implement safeguards, since we wanted to maximise speed of development. The times that we did break everything, we think would not have been prevented by the safeguards we had.
Now that we have something more or less final, it would be nice knowing that the main branch is guarded a bit more. We also discussed that it would be awesome to use github actions to make a small webpage for the project.

An example of a triggered pytest workflow testing out several OS can be seen here: https://github.com/dhsvendsen/MLOps-Project-Detox/actions/runs/3961690366


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used hydra with a default.yaml config until the final day of the project. This seemed to provide a pretty neat structure. Then we got sufficiently tired of two things: For one the increased clutter in the code, but mainly the automatic change of working directory to an output/day-hour-etc directory, which made managing paths cumbersome. In the end we made a single config.json file which solved our problems. We all agreed that we did not miss hydra.


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

Our main experiment consisted in doing a W&B sweep over a set of hyperparameters.  This can be thought of as running experiments, since we are experimentally (using Bayesian optimization) determining which combination of hyperparameters leads to the best performance and how good this performance is.
The W&B sweep is not directly reproducible, since the code is set up to write to our W&B project. However, this can easily be changed. Additionally, by showing our W&B report, we are able to document the performance of our model.


Regarding config files, we list the model hyperparameters in a config.json file which is loaded when training the model. This ensures that when training the final model, we can set the optimal hyperparameters identified through the W&B sweep, and then log the training and validation loss again using W&B. 


### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We have developped two docker images for our project: one for training the model (INSERT LINK) and a second for deploying the model to a FastApi (INSERT LINK).

You can work with these images locally or through GCP.
      
To work locally you can take the dockerfiles from our github (as described earlier) and build and run these. As an example here is how you would use the docker image for deployment/FastApi locally:
```bash
docker build -f simple_app.dockerfile . -t fastapi:latest
```
```bash
docker run -p 8080:8000 fastapi:latest
```
afterwards you can access the fastapi on localhost:8080
      
To run everything on GCP, you can use the existing image in our container registry to deploy the FastApi through `Cloud Run`:

```bash
gcloud run deploy fastapi_deployment --image gcr.io/modified-wonder-374308/test_app:latest    JESPER INSERT PATH HERE --platform managed --region europe-west1 --allow-unauthenticated --port 8080
```
Afterwards the service will be available at the endpoint which `Cloud Run` returns.




### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

We used a powerful IDE (vscode or pycharm depending on the group member) to debug our code. By setting breakpoints we were able to track where the code fails and try out different fixes in the debugging console. A program like pycharm has an especially powerful debugger, since it gives an excellent overview of which values the different variables have at any given time, as well allowing one to jump between objects by ctrl+left-click’ing them.


Regarding profiling, we elected to prioritize other tasks and therefore did not profile our code. This means that performance was probably left on the table. However, when using the pytorch lightning module, it alerts the user to potential bottlenecks such as the number of workers in the dataloaders etc. Some of these suggested changes did in fact speed up the code, although we were not able to document it.


## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

- `Compute Engine` for training our model. Specifically we spin up a VM with our training docker image. Here we pull data from a bucket (using `dvc pull`), run the model, and export a checkpoint to a bucket.
- `Container Registry` is used for storing images. We have two images that we use: an image for training our model and an image for deploying it through FastApi.
- `Buckets` is used for storing data and trained models (checkpoints). The data bucket is version controlled using dvc, which allows us to easily pull data from the bucket when we train the model.
- `Cloud Run` is used for deploying our FastApi. We use our deployment/FastApi docker image, and create a Cloud Run function from this, making it easier to do inference for anyone interested in using our application.
- `Cloud Build` is used for building the docker images. Specifically we have created a trigger such that our `CloudBuild.yaml`is executed when ever we push to our main branch in the GitHub Repo. This starts the Cloud Build service and we get a new image in our container registry for both the training image and deployment/FastApi image.



### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

For our training flow, we used google compute engine (GCE) to launch a virtual machine (VM) based on the Container Optimised Image (COS). COS is a lightweight OS designed specifically for running docker containers on. We configured the VM to use 1 x V100 GPU to accelerate training of our model, which otherwise took more than 5 minutes per epoch to run on our local CPU. We decided to run training on a dedicated VM, to make it easier for us to troubleshoot by being able to SSH into the machine, unlike Cloud Run. Configuring the VM turned out to be more difficult than expected: COS comes without GPU support natively, and requires installing dependencies; there is a simple google command to do so, but it turns out there is a known bug that makes the simple command non functional with the latest released COS. Hence one manually has to specify the VM to use the second-to-latest release of COS instantiating the instance, which can only be done through the CLI. However, it should be possible to combine all these CLI commands into a single makefile, so that the user could simply call “Make Cloud Train” for example.


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 22 fill here ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not manage to implement monitoring. Implementing monitoring could potentially help the longevity of our application by alerting us of model decay and data drift. 

In all machine learning models it is fair to expect that there might be some change to the data and as such to a models performance if it is not retrained. In our specific case, we are working with user generated comments from wikipedia, and rating their toxicity levels. Especially within this context we might expect the user behaviour to change over time. As an example, acrononyms with different sentiments migth change over time (ie. "WTF" or "xoxo"), causing model decay, as it would not be trained on such examples. Furthermore, the concept of toxicity might change, and create a need for retraining the models on data with adjusted labels. 

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 26 fill here ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---

