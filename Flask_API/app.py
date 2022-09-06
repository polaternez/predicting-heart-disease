from hashlib import new
import json
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


app = Flask(__name__)
model = pickle.load(open("models/final_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Age = float(request.form["Age"])
    Sex = request.form["Sex"]
    ChestPainType = request.form["ChestPainType"]
    RestingBP = float(request.form["RestingBP"])
    Cholesterol = float(request.form["Cholesterol"])
    FastingBS = float(request.form["FastingBS"])
    RestingECG = request.form["RestingECG"]
    MaxHR = float(request.form["MaxHR"])
    ExerciseAngina = request.form["ExerciseAngina"]
    Oldpeak = float(request.form["Oldpeak"])
    ST_Slope = request.form["ST_Slope"]
    new_data = np.array([Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
                         RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope])
    features = pd.DataFrame(
        np.array(new_data).reshape(-1, 11),
        columns=['Age', 'Sex', 'ChestPainType', 'RestingBP','Cholesterol', 'FastingBS',
                'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    )
    prediction = model.predict(features)
    
    output = ""
    if prediction[0]==0:
        output = "Patient looks healthy!!"
    elif prediction[0]==1:
        output = "Patient probably has heart disease!!"
    else:
        output = "Error!!"

    return render_template("index.html", prediction_text=output)
        
@app.route("/predict_api", methods=["POST"])
def predict_api():
    '''
    For direct API calls trought request
    '''
    request_json = request.get_json()
    new_data = request_json["input"]
    features = pd.DataFrame(
        np.array(new_data).reshape(-1, 11),
        columns=['Age', 'Sex', 'ChestPainType', 'RestingBP','Cholesterol', 'FastingBS',
                'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    )
    prediction = model.predict(features)
    output = prediction[0]
    
    response = json.dumps({'response': str(output)})
    return response, 200

if __name__ == "__main__":
    app.run(debug=True)


