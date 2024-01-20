# Heart Disease Predictor: Project Overview  
This tool was created to help people by predicting heart disease early
* Take Heart Failure Prediction Dataset from Kaggle.
* Exploratory Data Analysis (EDA)
* Build transformation pipeline
* Train different models and evaluate them using cross validation.
* Built a client facing API using Flask 

Note: This project was made for educational purposes.

## Code and Resources Used 
**Python Version:** 3.8  
**Packages:** numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, flask, json, pickle  
**For Dataset:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction 

**Create Anaconda environment:** 
- ```conda create -p venv python==3.8 -y```  
- ```pip install -r requirements.txt```
  

## Data Cleaning
We take data from Kaggle. This data has already been cleared!

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables. 

![alt text](https://github.com/polaternez/predicting_heart_disease/blob/master/reports/figures/HeartDisease_counts.jpg "Heart Disease Counts")
![alt text](https://github.com/polaternez/predicting_heart_disease/blob/master/reports/figures/Sex.jpg "Heart Disease by Sex")
![alt text](https://github.com/polaternez/predicting_heart_disease/blob/master/reports/figures/age.jpg "Heart Disease by Age")
![alt text](https://github.com/polaternez/predicting_heart_disease/blob/master/reports/figures/correlation.jpg "Correlation")

## Model Building 

First, we split the data into train and test sets with a test size of 20%. After that create transformation pipeline with this pipeline, encode categorical features 
and standardize numeric features.   

We try five distinct models and evaluate them using accuracy. Then we get the following results:

![alt text](https://github.com/polaternez/predicting_heart_disease/blob/master/reports/figures/model_performance.png "Model Performances")

Thus, we choose the Random Forest model, it has the highest accuracy, then fine-tune model for better performance.

## Productionization 
In this step, I created the UI with the Flask. API endpoint help receives a request and returns the results of the heart disease classification.

![alt text](https://github.com/polaternez/predicting_heart_disease/blob/master/reports/figures/heart_disease_api.png "Predict Heart Disease")



