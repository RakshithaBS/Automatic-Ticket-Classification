# Automatic-Ticket-Classification

## Topics Covered
* Overview
* Motivation
* Technical Aspects
* Setup
* Demo


## Overview

The project focuses on building a system which classifies the customer complaint into irrespective topics/categories. The project emphasises on two major tasks:
1. Topic modelling
2. Multi label classification
The final system is a classified model which is used to predict the class/product category of the input customer complaint. The dataset is in the json format which consists of customer complaints. The complaints have to catgorised into five categories :
* Credit card / Prepaid card
* Bank account services
* Theft/Dispute reporting
* Mortgages/loans
* Others
After pre-processing ,EDA ,topic modelling and model training, the model is deployed using a flask application.


## Motivation

For a financial company, customer complaints carry a lot of importance, as they are often an indicator of the shortcomings in their products and services. If these complaints are resolved efficiently in time, they can bring down customer dissatisfaction to a minimum and retain them with stronger loyalty. This also gives them an idea of how to continuously improve their services to attract more customers.
These customer complaints are unstructured text data; so, traditionally, companies need to allocate the task of evaluating and assigning each ticket to the relevant department to multiple support employees. This becomes tedious as the company grows and has a large customer base.
The goal of the project is to build an ML system which classifies customer complaints based on the products/services. By doing so, you can segregate these tickets into their relevant categories and, therefore, help in the quick resolution of the issue.

## Technical Aspects

### Data Pre-processing 
* The initial step was to convert the JSON data into a pandas dataframe, which contained 78k customer complaints. Following this, some basic text cleaning was performed, such as handling missing data, converting the text to lowercase, removing digits and punctuation, and eliminating masked characters and text in square brackets.
* Since the objective was to categorize the text into five products, the focus was on identifying noun words as they play a crucial role in this task. Thus, only the noun words were filtered and used for predictive modeling.
* The final step involved applying a td_idf transformation to the data before feeding it into the NMF model, which produced the desired results.

### Topic modelling using NMF
Non-negative matrix factorization (NMF) is a matrix decomposition method that can be used for a variety of applications, including topic modeling. NMF factorizes a non-negative data matrix into two non-negative matrices, where the resulting matrices can be interpreted as the topic-term matrix and the document-topic matrix.This project uses sklearn module to implement NMF.

### Mapping the predictions to five categories
The NMF model transformation gives the output as numbers between 0-4. This has to me mapped to respective categories based on intuition.

### Multi-Label classification 
Two models, logistic regression with hyperparameter tuning and XGBoost, were evaluated in the project. Surprisingly, logistic regression outperformed XGBoost on the given dataset, and was ultimately chosen as the final model.

The notebooks for the above sections is covered here [PreProcessingAndModelTraining](https://github.com/RakshithaBS/Automatic-Ticket-Classification/blob/master/Automatic_Ticket_classification_Case_Study.ipynb)
### Deploying the model using flask
A simple web service was built using flask which exposes an api ```/predict``` to  make the model predictions.

## Setup
1. Install necessary packages
Please run this command to install the necessary packages to run the project.

```
pip install -r requirements.txt
```
2. Download the en_core_web_sm model

```
python -m spacy download en_core_web_sm
```
3. Run the flask application

```
python app.py
```
The script to run flask application [Flask_application](https://github.com/RakshithaBS/Automatic-Ticket-Classification/blob/master/app.py)

## Demo

The notebook here [demo_notebook](https://github.com/RakshithaBS/Automatic-Ticket-Classification/blob/master/inference.ipynb) demonstrates calling the predict api to make predicts.
