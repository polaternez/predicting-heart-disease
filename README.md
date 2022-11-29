# Heart Disease Predictor: Project Overview  
* This tool was created to help people by predicting heart disease early
* Take Heart Failure Prediction Dataset from Kaggle.
* Exploratory Data Analysis (EDA)
* Build transformation pipeline
* Train different models and evaluate them using cross validation.
* Built a client facing API using Flask 

Note: This project was made for educational purposes.

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** numpy, pandas, matplotlib, seaborn, sklearn, xgboost, flask, json, pickle  
**For Flask API Requirements:**  ```pip install -r requirements.txt```  
**For Dataset:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction 

## Data Cleaning
We take data from Kaggle. This data has already been cleared!

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables. 

![alt text](https://github.com/polaternez/heart_disease_proj/blob/data_eda/images/HeartDisease_counts.jpg "Heart Disease Counts")
![alt text](https://github.com/polaternez/heart_disease_proj/blob/data_eda/images/Sex.jpg "Heart Disease by Sex")
![alt text](https://github.com/polaternez/heart_disease_proj/blob/data_eda/images/age.jpg "Heart Disease by Age")
![alt text](https://github.com/polaternez/heart_disease_proj/blob/data_eda/images/correlation.jpg "Correlation")

## Model Building 

First, we split the data into train and test sets with a test size of 20%. After that create transformation pipeline with this pipeline, encode categorical features 
and standardize numeric features.   

We try five distinct models and evaluate them using accuracy. Then we get the following results:

![alt text](https://github.com/polaternez/heart_disease_proj/blob/data_eda/images/model_performance.png "Model Performances")

Thus, we choose the Random Forest model, it has the highest accuracy, then fine-tune model for better performance.

## Productionization 
In this step, I created the UI with the Flask. API endpoint help receives a request and returns the results of the heart disease classification.




