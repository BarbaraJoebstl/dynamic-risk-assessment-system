


Project Steps Overview
You'll complete the project by proceeding through 5 steps:

1. Data ingestion. ✅
Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.

2. Training, scoring, and deploying. ✅
Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.


3. Diagnostics. ✅
Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.


4. Reporting. ✅
Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.


5. Process Automation. ✅
Create a script and cron job that automatically run all previous steps at regular intervals.



# About the starter code
/practicedata/. This is a directory that contains some data you can use for practice.
/sourcedata/. This is a directory that contains data that you'll load to train your models.
/ingesteddata/. This is a directory that will contain the compiled datasets after your ingestion script.
/testdata/. This directory contains data you can use for testing your models.
/models/. This is a directory that will contain ML models that you create for production.
/practicemodels/. This is a directory that will contain ML models that you create as practice.
/production_deployment/. This is a directory that will contain your final, deployed models.

The following are the Python files that are in the starter files:

training.py, a Python script meant to train an ML model
scoring.py, a Python script meant to score an ML model
deployment.py, a Python script meant to deploy a trained ML model
ingestion.py, a Python script meant to ingest new data
diagnostics.py, a Python script meant to measure model and data diagnostics
reporting.py, a Python script meant to generate reports about model metrics
app.py, a Python script meant to contain API endpoints
wsgi.py, a Python script to help with API deployment
apicalls.py, a Python script meant to call your API endpoints
fullprocess.py, a script meant to determine whether a model needs to be re-deployed, and to call all other Python scripts when needed
