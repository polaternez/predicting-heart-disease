# Heart Disease Predictor: Project Overview  
This project aims to facilitate early prediction of heart disease to aid in proactive healthcare management.

- Utilizes Heart Failure Prediction Dataset from Kaggle.
- Performs Exploratory Data Analysis (EDA) for initial insights.
- Builds a robust transformation pipeline for data preparation.
- Trains and evaluates various machine learning models using cross-validation.
- Deploys a user-friendly API using Flask for heart disease prediction.

Note: This project was developed for educational purposes


## Code and Resources
**Python Version:** 3.8  
**Packages:** numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, flask, json, pickle  
**Setting Up Environment:** 
- ```conda create -p venv python=3.8 -y```  
- ```pip install -r requirements.txt```

**Dataset:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction 


## Getting Data
The project utilizes the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction), obtained from Kaggle. This dataset boasts a unique origin, combining five previously independent heart disease datasets that share 11 common features. Following the merging process, duplicate observations were removed, resulting in a final dataset of 918 observations.

## EDA
The EDA examined data distributions and value counts for categorical variables. Key insights from pivot tables are visualized in the following figures:

![alt text](https://github.com/polaternez/predicting_heart_disease/blob/master/reports/figures/HeartDisease_counts.jpg "Heart Disease Counts")
![alt text](https://github.com/polaternez/predicting_heart_disease/blob/master/reports/figures/Sex.jpg "Heart Disease by Sex")
![alt text](https://github.com/polaternez/predicting_heart_disease/blob/master/reports/figures/age.jpg "Heart Disease by Age")
![alt text](https://github.com/polaternez/predicting_heart_disease/blob/master/reports/figures/correlation.jpg "Correlation")


## Model Building 
- Split the data into train and test sets with a test size of 20%
- Constructed a data transformation pipeline that encodes categorical features and standardizes numerical features.
- Leveraging cross-validation, we evaluated multiple models to predict heart disease, prioritizing both accuracy and training efficiency. Ultimately, the Random Forest model emerged as the optimal choice due to its superior performance.
- Fine-tune the Random Forest model to achieve optimal performance.
  
The results of cross-validation for the models are as follows:

![alt text](https://github.com/polaternez/predicting_heart_disease/blob/master/reports/figures/model_performance.png "Model Performances")


## Productionization 
Deployed a user-friendly API using Flask. The API endpoint accepts user requests and returns the predicted heart disease classification.

![alt text](https://github.com/polaternez/predicting_heart_disease/blob/master/reports/figures/heart_disease_api.png "Predict Heart Disease")



